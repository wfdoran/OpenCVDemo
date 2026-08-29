"""
pollen_detector.py

A Limelight "snapscript" (Python vision pipeline) that finds Pollen game
pieces (yellow plastic wiffle balls, used in the FTC BioBuzz game) in a
camera image.

HOW IT WORKS, IN PLAIN ENGLISH
-------------------------------
1.  We look at every pixel and decide whether it is "yellow enough" to
    maybe be part of a pollen ball.  This gives us a black-and-white
    mask image where white = "probably yellow."
2.  We group the white pixels into separate blobs (one blob per ball,
    roughly) and throw out blobs that are way too small, way too big,
    or the wrong shape to be a ball.
3.  For each remaining blob we trace its outline (its "contour") and
    feed those outline points into a RANSAC circle-fitting algorithm.
    RANSAC is important here because a pollen ball is a wiffle ball —
    it is covered in holes, so its outline is not a perfect circle.
    It has little notches and bites taken out of it wherever a hole
    happens to sit near the edge of the ball.  A simple "best fit"
    circle would get dragged off-center by those notches.  RANSAC
    instead keeps randomly guessing circles from 3 outline points at a
    time, counts how many of the OTHER outline points agree with each
    guess, and keeps the guess that the most points agree with.  The
    notches end up out-voted by the much larger number of points that
    really do lie on the ball's true round edge.
4.  Circles that survive the RANSAC vote (enough outline points agreed,
    and the size is ball-sized) get drawn on the output image and
    reported back to the robot.

This file is meant to run two ways:
  * On the Limelight camera itself, which repeatedly calls
    runPipeline(image, llrobot) on every video frame.
  * On a regular computer, by running
        python pollen_detector.py some_image.jpg
    which loads one image, runs the exact same runPipeline() function,
    and pops up a window showing what was detected.
"""

import math
import random
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# TUNABLE CONSTANTS
#
# These numbers were chosen by looking at real photos of pollen balls
# (pollen_data/images) and measuring how yellow the balls are and how
# big they are in pixels.  If pollen balls look different in your own
# photos (different lighting, different camera, closer/farther away),
# these are the first numbers to adjust.
# ---------------------------------------------------------------------------

# --- Color filter (HSV = Hue, Saturation, Value) ---
# Hue is "what color" (0-179 in OpenCV), Saturation is "how vivid/pure
# the color is" (0 = gray, 255 = fully saturated), and Value is
# "how bright" (0 = black, 255 = full brightness).  Pollen-ball yellow
# sits in a narrow hue band, so filtering on hue is a reliable way to
# ignore the gray floor, red walls, and other colored game pieces.
YELLOW_HUE_LOW = 12
YELLOW_HUE_HIGH = 38
YELLOW_SAT_MIN = 80
YELLOW_VAL_MIN = 80

# --- Blob size/shape filters ---
# A "blob" is one connected clump of yellow pixels.  We only bother
# trying to fit a circle to a blob if its bounding box is roughly
# ball-shaped and ball-sized in pixels.
MIN_BLOB_SIDE_PX = 14      # smaller than this is probably noise
MAX_BLOB_SIDE_PX = 320     # bigger than this is probably a whole wall
MIN_BLOB_ASPECT = 0.45     # width/height (or height/width) can't be too skinny
MIN_BLOB_FILL_RATIO = 0.35  # (yellow pixels) / (bounding box area)

# --- RANSAC circle fit ---
RANSAC_ITERATIONS = 350          # how many random 3-point guesses to try
RANSAC_INLIER_DISTANCE_PX = 3.0  # how close (px) a point must be to "agree"
RANSAC_MIN_INLIER_RATIO = 0.35   # fraction of outline points that must agree
RANSAC_MIN_INLIERS = 25          # absolute minimum number of agreeing points
MIN_BALL_RADIUS_PX = 8
MAX_BALL_RADIUS_PX = 160
MAX_POINTS_FOR_RANSAC = 350      # cap for speed on slower hardware (Limelight)
MAX_CIRCLES_PER_BLOB = 2         # handles two balls touching/overlapping

# --- Final sanity check on each RANSAC circle ---
# A real ball's mask blob is a solid filled disk, not just a ring of
# outline points. This catches the rare case where 3 randomly-chosen
# outline points happen to roughly agree on a circle that doesn't
# actually correspond to a solid yellow disk (this shows up most often
# on skin tones, which can be similar in hue to pollen yellow but form
# thin/irregular blobs rather than solid round ones).
MIN_DISK_FILL_RATIO = 0.75

# --- Drawing ---
CIRCLE_COLOR_BGR = (0, 255, 0)     # green
CENTER_DOT_COLOR_BGR = (0, 0, 255)  # red
LINE_THICKNESS = 2


# ---------------------------------------------------------------------------
# STEP 1: Color segmentation
# ---------------------------------------------------------------------------
def get_yellow_mask(image):
    """
    Builds a black-and-white "mask" image that is white wherever a pixel
    looks like pollen-ball yellow, and black everywhere else.

    We first convert the image from BGR (Blue, Green, Red — OpenCV's
    default) into HSV, because HSV separates "what color" (hue) from
    "how bright/vivid" (value/saturation).  That makes it much easier
    to write one simple rule ("hue is roughly yellow, and it's a fairly
    pure, fairly bright yellow") that keeps working across different
    lighting instead of having to reason about three tangled BGR values.

    After thresholding, the mask usually has small stray white specks
    (camera noise, tiny yellow reflections) and small black speckles
    inside the ball blobs (where a bright highlight briefly looks less
    saturated).  A "morphological open" operation cleans this up: it
    is equivalent to eroding away anything too thin to survive, then
    growing what's left back to its original size, which erases
    isolated specks without changing the overall shape of real blobs.
    We deliberately do NOT try to fill in the wiffle-ball holes here —
    those become the "outliers" that the RANSAC step is designed to
    ignore, which is the whole point of using RANSAC instead of a
    simpler circle fit.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([YELLOW_HUE_LOW, YELLOW_SAT_MIN, YELLOW_VAL_MIN])
    upper_yellow = np.array([YELLOW_HUE_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


# ---------------------------------------------------------------------------
# STEP 2: Turn the mask into a list of candidate ball-shaped blobs
# ---------------------------------------------------------------------------
def find_candidate_blobs(mask):
    """
    Finds the outlines ("contours") of every separate white blob in the
    mask, and throws away any blob that clearly can't be a pollen ball
    before we bother running the (more expensive) RANSAC circle fit on
    it. This keeps the pipeline fast enough to run on every camera frame.

    We check three things about each blob's bounding box:
      * Size — not too tiny (probably noise) and not huge (probably a
        big yellow-ish patch of floor or wall, not a single ball).
      * Aspect ratio — a ball's bounding box is roughly square, so a
        long skinny blob (like a stray reflection along an edge) is
        rejected.
      * Fill ratio — how much of the bounding box is actually yellow.
        A real ball fills a big circular fraction of its box; a thin
        crescent-shaped glare wouldn't.

    Returns a list of contours (each contour is the list of (x, y)
    points that trace one blob's outline).
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_BLOB_SIDE_PX or h < MIN_BLOB_SIDE_PX:
            continue
        if w > MAX_BLOB_SIDE_PX or h > MAX_BLOB_SIDE_PX:
            continue

        aspect = min(w, h) / float(max(w, h))
        if aspect < MIN_BLOB_ASPECT:
            continue

        blob_area = cv2.contourArea(contour)
        box_area = float(w * h)
        if box_area <= 0 or (blob_area / box_area) < MIN_BLOB_FILL_RATIO:
            continue

        candidates.append(contour)

    return candidates


# ---------------------------------------------------------------------------
# STEP 3: RANSAC circle fitting
# ---------------------------------------------------------------------------
def circle_from_three_points(p1, p2, p3):
    """
    Given exactly three (x, y) points, computes the one circle that
    passes through all three of them (this is the "minimal model" a
    RANSAC circle fit is built from — a circle has 3 degrees of
    freedom, cx/cy/r, so it takes exactly 3 points to pin one down).

    Returns (cx, cy, r), or None if the three points are (nearly)
    collinear, in which case no finite circle passes through them.
    """
    (ax, ay), (bx, by), (cx3, cy3) = p1, p2, p3

    d = 2.0 * (ax * (by - cy3) + bx * (cy3 - ay) + cx3 * (ay - by))
    if abs(d) < 1e-6:
        return None  # points are (nearly) on a straight line

    ax2ay2 = ax * ax + ay * ay
    bx2by2 = bx * bx + by * by
    cx2cy2 = cx3 * cx3 + cy3 * cy3

    ux = (ax2ay2 * (by - cy3) + bx2by2 * (cy3 - ay) + cx2cy2 * (ay - by)) / d
    uy = (ax2ay2 * (cx3 - bx) + bx2by2 * (ax - cx3) + cx2cy2 * (bx - ax)) / d

    r = math.hypot(ux - ax, uy - ay)
    return ux, uy, r


def least_squares_circle_fit(points):
    """
    Given a set of (x, y) points that are already believed to lie on
    (or very near) one circle, computes the single best-fitting circle
    through all of them at once, using linear least squares.

    Why bother, if RANSAC already gave us a circle? RANSAC's circle
    came from just 3 points, which is enough to find roughly the right
    circle but is still sensitive to exactly which 3 points got picked.
    Once RANSAC has told us WHICH points belong to the ball's edge (the
    "inliers"), we get a noticeably more accurate and stable center and
    radius by fitting one smooth circle through all of those points
    together, instead of trusting just 3 of them.

    The math: a circle x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    is linear in the unknowns (cx, cy, k) where k = r^2 - cx^2 - cy^2,
    so we can solve for them with numpy's least-squares solver instead
    of an iterative search.
    """
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y

    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = solution
    r_squared = k + cx * cx + cy * cy
    if r_squared <= 0:
        return None
    return cx, cy, math.sqrt(r_squared)


def ransac_fit_circle(points, rng):
    """
    The core RANSAC ("RANdom SAmple Consensus") circle detector.

    The idea, step by step:
      1. Randomly pick 3 of the outline points and compute the one
         circle that passes exactly through them (a "hypothesis").
      2. Check every other outline point: is it close to that
         hypothesis circle's edge (within RANSAC_INLIER_DISTANCE_PX)?
         Points that are close are called "inliers" and are treated as
         votes for that hypothesis; points that are far away (for us,
         mainly points along a wiffle-ball hole notch, or a second
         overlapping ball) are "outliers" and are ignored for this
         hypothesis.
      3. Remember whichever hypothesis got the most votes.
      4. Repeat many times, because most random triples will land on
         at least one outlier and produce a bad/wrong circle — we're
         relying on the fact that, over enough random tries, we will
         eventually pick 3 points that are all genuinely on the ball's
         true edge, and that hypothesis will beat all the others.

    This is exactly why RANSAC tolerates noisy, imperfect outlines so
    much better than a plain average/least-squares fit would: a plain
    fit uses ALL the points every time (so outliers always drag it off
    target), while RANSAC actively searches for the hypothesis that
    the majority of points agree with, and only trusts that majority.

    Returns (cx, cy, r, inlier_points) for the best circle found, or
    None if nothing good enough was found.
    """
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    if n_points < 3:
        return None

    best_inliers = None
    best_count = -1

    for _ in range(RANSAC_ITERATIONS):
        sample_idx = rng.sample(range(n_points), 3)
        hypothesis = circle_from_three_points(*points[sample_idx])
        if hypothesis is None:
            continue

        cx, cy, r = hypothesis
        if not (MIN_BALL_RADIUS_PX <= r <= MAX_BALL_RADIUS_PX):
            continue  # not a ball-sized circle; skip without scoring it

        # Distance from every point to the CENTER, minus the radius,
        # tells us how far that point is from the circle's EDGE.
        dist_to_center = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        dist_to_edge = np.abs(dist_to_center - r)
        inlier_mask = dist_to_edge <= RANSAC_INLIER_DISTANCE_PX
        count = int(np.count_nonzero(inlier_mask))

        if count > best_count:
            best_count = count
            best_inliers = points[inlier_mask]

    if best_inliers is None:
        return None

    inlier_ratio = best_count / float(n_points)
    if best_count < RANSAC_MIN_INLIERS or inlier_ratio < RANSAC_MIN_INLIER_RATIO:
        return None

    # Polish the winning hypothesis using ALL of its inliers at once.
    refined = least_squares_circle_fit(best_inliers)
    if refined is None:
        return None
    cx, cy, r = refined
    if not (MIN_BALL_RADIUS_PX <= r <= MAX_BALL_RADIUS_PX):
        return None

    return cx, cy, r, best_inliers


def circle_disk_fill_ratio(mask, cx, cy, r):
    """
    Checks whether the DISK (the filled-in interior, not just the
    outline) that a candidate circle covers is actually, mostly,
    yellow according to the mask.

    RANSAC only looks at outline points, so it is possible — though
    rare — for 3 randomly-chosen points to roughly agree on a circle
    that doesn't really match a solid ball. This is a cheap final
    sanity check: crop out the small square around the candidate
    circle, and directly measure what fraction of the mask pixels
    inside the circle are actually white (yellow). A real pollen ball
    should score close to 1.0 here; a false alarm typically scores
    much lower because its "circle" doesn't correspond to a solid
    filled blob in the mask.
    """
    height, width = mask.shape[:2]
    x0 = max(0, int(cx - r))
    y0 = max(0, int(cy - r))
    x1 = min(width, int(cx + r) + 1)
    y1 = min(height, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside_disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    pixels_inside = int(np.count_nonzero(inside_disk))
    if pixels_inside == 0:
        return 0.0

    mask_patch = mask[y0:y1, x0:x1]
    yellow_inside = int(np.count_nonzero(mask_patch[inside_disk]))
    return yellow_inside / float(pixels_inside)


def detect_circles_in_blob(contour, mask, rng):
    """
    Runs RANSAC circle fitting on one blob's outline, and returns a
    list of (cx, cy, r) circles found in it (usually one, but see
    below).

    Most blobs are a single ball, so this normally finds one circle
    and stops. But sometimes two pollen balls are touching or
    overlapping in the image, which the color mask sees as a single,
    peanut-shaped blob. To handle that gracefully, after finding a
    circle we remove its inlier points from the pool and try RANSAC
    again on whatever outline points are left over; if a second good
    circle turns up (and its center isn't basically on top of the
    first one), we keep both. We cap this at MAX_CIRCLES_PER_BLOB
    circles per blob so we don't chase noise forever.

    `mask` (the full yellow mask image) is passed in only so each
    candidate circle can be double-checked with circle_disk_fill_ratio()
    before we accept it.
    """
    contour_points = contour.reshape(-1, 2).astype(np.float64)

    # Large blobs can have hundreds of outline pixels. RANSAC's
    # accuracy doesn't need every single one, so on hardware like the
    # Limelight (which must keep up with live video) we randomly
    # subsample down to a manageable number of points to keep each
    # frame fast, without meaningfully hurting the fit quality.
    if len(contour_points) > MAX_POINTS_FOR_RANSAC:
        keep_idx = rng.sample(range(len(contour_points)), MAX_POINTS_FOR_RANSAC)
        contour_points = contour_points[keep_idx]

    found_circles = []
    remaining_points = contour_points

    for _ in range(MAX_CIRCLES_PER_BLOB):
        if len(remaining_points) < RANSAC_MIN_INLIERS:
            break

        result = ransac_fit_circle(remaining_points, rng)
        if result is None:
            break

        cx, cy, r, inliers = result

        if circle_disk_fill_ratio(mask, cx, cy, r) < MIN_DISK_FILL_RATIO:
            # The outline points agreed on a circle, but the inside of
            # that circle isn't actually a solid yellow disk, so this
            # blob probably isn't a ball after all. No point trying
            # again on the leftover points of a non-ball blob.
            break

        # Skip near-duplicate circles (this can happen if leftover
        # points after removing the first circle's inliers still
        # loosely describe roughly the same circle).
        is_duplicate = any(
            math.hypot(cx - fx, cy - fy) < 0.5 * max(r, fr)
            for fx, fy, fr in found_circles
        )
        if not is_duplicate:
            found_circles.append((cx, cy, r))

        # Remove this circle's inlier points before trying again, so
        # a second RANSAC pass is forced to explain the LEFTOVER
        # points instead of just rediscovering the same circle.
        inlier_set = {tuple(p) for p in inliers}
        remaining_points = np.array(
            [p for p in remaining_points if tuple(p) not in inlier_set]
        )

    return found_circles


# ---------------------------------------------------------------------------
# STEP 4: Drawing + packaging results
# ---------------------------------------------------------------------------
def draw_detections(image, circles):
    """
    Draws a green circle outline plus a small red center dot on top of
    every detected pollen ball, so a human looking at the output image
    can immediately see what the pipeline found. Drawing happens on
    the image that gets returned to the caller; it never changes the
    pixels that were analyzed, so it can't accidentally bias detection.
    """
    for cx, cy, r in circles:
        center = (int(round(cx)), int(round(cy)))
        radius = int(round(r))
        cv2.circle(image, center, radius, CIRCLE_COLOR_BGR, LINE_THICKNESS)
        cv2.circle(image, center, 3, CENTER_DOT_COLOR_BGR, -1)
    return image


def build_llpython(circles):
    """
    Packs the detection results into the simple list of numbers
    ("llpython") that the Limelight sends to the robot over
    NetworkTables. Robot code can't easily read a variable-length list,
    so — following the usual Limelight convention — we always return a
    fixed-length list: how many pollen balls were found, followed by
    the (x, y, radius) of up to the two largest ones, zero-padded if
    fewer than two were found.

    circles is sorted largest-first so that if the robot only cares
    about "the biggest/closest pollen," it's always llpython[1:4].
    """
    circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)

    llpython = [float(len(circles_sorted))]
    for i in range(2):
        if i < len(circles_sorted):
            cx, cy, r = circles_sorted[i]
            llpython.extend([float(cx), float(cy), float(r)])
        else:
            llpython.extend([0.0, 0.0, 0.0])

    return llpython


# ---------------------------------------------------------------------------
# THE LIMELIGHT ENTRY POINT
# ---------------------------------------------------------------------------
def runPipeline(image, llrobot):
    """
    This is the function the Limelight camera calls, by name, on every
    single video frame. It must follow the Limelight snapscript
    contract exactly:

        Input:
            image   - a BGR image (numpy array) straight from the camera
            llrobot - a list of numbers the robot sent us (unused here,
                      since the prompt says to assume it is empty)

        Output (in this exact order):
            largestContour - a contour the Limelight can also use for
                              its built-in crosshair/targeting tools.
                              We don't need that feature for this
                              project, so we return an empty array,
                              which tells the Limelight "no contour."
            image           - the (possibly annotated) image to show on
                              the Limelight's web dashboard.
            llpython        - our own list of numbers for the robot to
                              read back over NetworkTables.

    Everything this function does is just calling the helper functions
    above in order: find the yellow blobs, fit a robust circle to each
    one with RANSAC, draw the results, and package them up.
    """
    # A fresh Random instance keeps RANSAC's randomness self-contained
    # (it doesn't depend on, or interfere with, any other random-number
    # use elsewhere in the robot code).
    rng = random.Random(0)

    output_image = image.copy()

    mask = get_yellow_mask(image)
    candidate_blobs = find_candidate_blobs(mask)

    detected_circles = []
    for blob in candidate_blobs:
        detected_circles.extend(detect_circles_in_blob(blob, mask, rng))

    draw_detections(output_image, detected_circles)

    llpython = build_llpython(detected_circles)
    largestContour = np.array([])

    return largestContour, output_image, llpython


# ---------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT (for testing on a laptop, off the robot)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pollen_detector.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    largestContour, annotated_image, llpython = runPipeline(input_image, [])

    num_found = int(llpython[0])
    print(f"Found {num_found} pollen ball(s).")
    for i in range(min(num_found, 2)):
        cx, cy, r = llpython[1 + 3 * i : 4 + 3 * i]
        print(f"  ball {i + 1}: center=({cx:.1f}, {cy:.1f})  radius={r:.1f}px")

    cv2.imshow("Pollen Detection", annotated_image)
    print("Press any key (with the image window focused) to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
