import cv2
import numpy as np
import sys

LOWER_YELLOW = np.array([15, 70, 100])
UPPER_YELLOW = np.array([42, 255, 255])

MIN_RADIUS = 10
MAX_RADIUS = 110
MIN_FILL_RATIO = 0.5  # fraction of circle area that must be yellow mask

# All of the numbers above (the blur size, the morphology kernel sizes, and
# especially MIN_RADIUS/MAX_RADIUS) were tuned by looking at the training
# photos, which are all 640 pixels wide. But a camera phone (like the ones
# used for the DiscordExamples photos) can take pictures that are 4032
# pixels wide - over 6 times bigger! At that size a pollen ball isn't 10-110
# pixels across anymore, it's 60-700+ pixels across, so MAX_RADIUS throws
# every real ball away and the whole detector comes up empty (or worse,
# latches onto some small unrelated yellow speck instead). Rather than
# retuning every constant for every possible camera resolution, we instead
# shrink any incoming image down to this same 640-pixel width before doing
# any detection work. That way the pollen balls always appear at roughly
# the same pixel size they did in the training photos, and all the tuned
# constants above keep working no matter what resolution the camera feed
# actually is.
DETECTION_WIDTH = 640


def _shrink_for_detection(image):
    # Resize the image down (never up) so its width matches DETECTION_WIDTH,
    # keeping the height proportional so nothing gets stretched or squished.
    # Returns the resized image plus the "scale" number used to do it, so the
    # caller can later multiply detected coordinates by 1/scale to translate
    # them back into the original, full-size image's coordinate system.
    height, width = image.shape[:2]
    scale = DETECTION_WIDTH / width
    if scale >= 1.0:
        # The image is already no wider than our detection resolution (this
        # is normally true for the original training/testing photos), so
        # there's nothing to shrink - use it as-is.
        return image, 1.0

    new_size = (DETECTION_WIDTH, int(round(height * scale)))
    # INTER_AREA is the interpolation method OpenCV recommends specifically
    # for shrinking images - it averages together the pixels being merged,
    # which looks much cleaner than other methods when making an image
    # smaller.
    small = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return small, scale


def _build_mask(image):
    # This function's job is to turn a normal color photo into a black-and-white
    # "mask" image, where every pixel that looks like pollen-yellow is painted
    # white (255) and every other pixel is painted black (0). Think of it like
    # tracing over the picture with a yellow highlighter, but only a computer's
    # idea of "yellow."
    #
    # Step 1: Colors in a normal image are stored as BGR (blue, green, red)
    # amounts, which is a bad format for picking out "yellow" because yellow-ish
    # colors can have wildly different BGR numbers depending on lighting. So we
    # convert to HSV instead: Hue (what color it is, like a position on a color
    # wheel), Saturation (how vivid/pure the color is, vs. washed out gray), and
    # Value (how bright it is, vs. dark). Yellow is easy to describe in HSV: a
    # narrow range of Hue, plus "fairly vivid" and "fairly bright."
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: cv2.inRange checks every pixel and asks "is this pixel's HSV value
    # between LOWER_YELLOW and UPPER_YELLOW?" If yes, it becomes white; if no,
    # it becomes black. The LOWER/UPPER numbers at the top of the file were
    # tuned by measuring the actual yellow pollen balls in sample photos, so the
    # range is snug enough to (mostly) avoid gray floor, red walls, and shadows.
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Step 3: The pollen balls are wiffle-ball-like spheres with lots of holes
    # in them, so the raw mask looks like white circles full of little black
    # holes (wherever a hole in the ball shows background instead of yellow
    # plastic). A "closing" operation (dilate, then erode) is a standard image-
    # processing trick that fills in small black gaps surrounded by white,
    # without changing the overall size/shape much. Here it patches over the
    # holes so each ball becomes one solid white blob instead of Swiss cheese.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: An "opening" operation (erode, then dilate) is the opposite trick
    # — it's good at erasing tiny stray white specks (noise) while leaving big
    # solid blobs alone. This cleans up any leftover one-or-two-pixel false
    # detections scattered around the image.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # The result is a clean black-and-white image: solid white blobs roughly
    # where the pollen balls are, black everywhere else.
    return mask


def _fill_ratio(mask, cx, cy, r):
    # This function answers one question: "If I draw a circle at (cx, cy) with
    # radius r, how much of that circle is actually covered by yellow pixels
    # in the mask?" It returns a fraction from 0.0 (none of the circle is
    # yellow) to 1.0 (the whole circle is yellow).
    #
    # Why do we need this? Later on, we'll use a shape-detection trick to guess
    # "here's a circle that might be a pollen ball" — but that guess could be
    # wrong (for example, a circle drawn mostly over gray floor with only a
    # sliver of yellow at the edge). This function is our sanity check: a real
    # pollen ball should be almost entirely covered by yellow, so a low fill
    # ratio is a red flag that the guessed circle probably isn't a real ball.

    h, w = mask.shape

    # Figure out the smallest rectangular box that fully contains the circle,
    # clipped so it never goes outside the actual image (you can't have a
    # pixel at x = -5 or beyond the image's width/height).
    x1, y1 = max(int(cx - r), 0), max(int(cy - r), 0)
    x2, y2 = min(int(cx + r) + 1, w), min(int(cy + r) + 1, h)
    if x2 <= x1 or y2 <= y1:
        # The circle is completely off-screen — there's nothing to measure.
        return 0.0

    # Draw a fresh, solid white circle (matching the requested center/radius)
    # onto a small blank black canvas the size of that bounding box. This
    # gives us a "perfect circle" template to compare against the real mask.
    circle_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx) - x1, int(cy) - y1), int(r), 255, -1)

    # Grab the matching little rectangle out of the real yellow mask.
    region = mask[y1:y2, x1:x2]

    circle_area = cv2.countNonZero(circle_mask)
    if circle_area == 0:
        return 0.0

    # bitwise_and keeps a pixel white only if it's white in BOTH images — i.e.,
    # both inside our perfect circle AND yellow in the real mask. Counting
    # those pixels and dividing by the circle's total area gives the fraction
    # of the circle that's actually yellow.
    overlap = cv2.countNonZero(cv2.bitwise_and(region, circle_mask))
    return overlap / circle_area


def _detect_balls(image):
    # This is the "brain" of the whole program. It takes a color photo and
    # returns a list of every pollen ball it thinks it found, where each ball
    # is described as (center_x, center_y, radius, confidence). It also
    # returns the yellow mask, in case the caller wants it too.

    # Shrink the photo down to our standard detection resolution first (see
    # the big comment on DETECTION_WIDTH above for why). "scale" tells us how
    # much smaller the working copy is than the original - for example, a
    # scale of 0.5 means the working copy is half the size, so a ball found
    # at radius 20 in the working copy is really radius 40 in the original.
    detection_image, scale = _shrink_for_detection(image)

    mask = _build_mask(detection_image)

    # Slightly blur the mask before circle-detection. This smooths out jagged,
    # noisy edges (like the bumpy outline left by the ball's holes) so the
    # circle-finding algorithm below sees cleaner, rounder shapes instead of
    # getting confused by every little bump.
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # cv2.HoughCircles implements a classic algorithm called the "Hough
    # Circle Transform." The intuition: for every "edge" pixel in the image
    # (a place where black meets white), imagine every possible circle of
    # every possible radius that could pass through that point, and cast a
    # "vote" for each one. After every edge pixel has voted, look for the
    # (center, radius) combinations that received a LOT of votes — those are
    # very likely to be real circles, because lots of different edge points
    # agreed that a circle belongs there. This is powerful because it can
    # find circles even when they're overlapping or partly hidden behind
    # something else, as long as enough of the curve is visible.
    #   dp=1.5      -> how detailed the internal "voting grid" is (a tradeoff
    #                  between accuracy and speed)
    #   minDist     -> circles whose centers are closer together than this are
    #                  treated as one candidate, to avoid a flood of near-
    #                  duplicate detections on the same ball
    #   param1      -> an edge-detection sensitivity setting used internally
    #   param2      -> the "how many votes does a circle need to count"
    #                  threshold; smaller finds more circles (but more false
    #                  alarms), bigger finds fewer (but misses faint ones)
    #   minRadius/maxRadius -> only look for circles in this size range, since
    #                  we already know roughly how big a pollen ball can look
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=MIN_RADIUS * 2,
        param1=100, param2=33, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
    )

    # HoughCircles is good but not perfect — it can occasionally propose a
    # circle that doesn't actually sit on a real yellow blob (for example, a
    # circle straddling the boundary between a ball and the floor). So we
    # double-check every candidate with _fill_ratio and throw out any circle
    # that isn't "mostly yellow inside."
    balls = []
    if circles is not None:
        for cx, cy, r in circles[0]:
            ratio = _fill_ratio(mask, cx, cy, r)
            if ratio >= MIN_FILL_RATIO:
                balls.append((float(cx), float(cy), float(r), ratio))

    # It's common for HoughCircles to report the SAME physical ball more than
    # once, as several slightly different (center, radius) guesses that all
    # overlap each other. We need to collapse those duplicates down to one
    # detection per real ball, while being careful NOT to throw away two
    # different balls that happen to be touching/overlapping each other in
    # the photo (which also produces circles with nearby centers!).
    #
    # The trick: sort candidates from biggest radius to smallest, and then go
    # through them one at a time, keeping a candidate only if its center is
    # NOT too close to a circle we've already kept. "Too close" is measured
    # relative to the two circles' radii (dist < 0.4 * (r1 + r2)) rather than
    # a fixed pixel distance, because bigger balls in the photo naturally
    # produce bigger circles with more legroom between their edges, while
    # small balls need a tighter distance check. The 0.4 cutoff was tuned by
    # testing: true duplicates of the same ball tend to land much closer
    # together than that (ratio around 0.2), while two real, different balls
    # that are partially stacked in the photo tend to land farther apart
    # (ratio around 0.44 or more) — so 0.4 sits in the gap between them.
    balls.sort(key=lambda b: b[2], reverse=True)
    kept = []
    for cx, cy, r, ratio in balls:
        overlaps = False
        for kcx, kcy, kr, _ in kept:
            dist = np.hypot(cx - kcx, cy - kcy)
            if dist < 0.4 * (r + kr):
                overlaps = True
                break
        if not overlaps:
            kept.append((cx, cy, r, ratio))

    # Put the biggest (most likely closest/most prominent) ball first, since
    # runPipeline() below treats the first entry as "the main target."
    kept.sort(key=lambda b: b[2], reverse=True)

    # Everything above was measured on the shrunken working copy, but the
    # caller needs coordinates that line up with the original, full-size
    # image (so the circles get drawn in the right place and the reported
    # positions make sense). Dividing by "scale" undoes the shrinking: if the
    # working copy was half-size (scale 0.5), multiplying its coordinates by
    # 1/0.5 = 2 converts them back to full-size coordinates.
    inverse_scale = 1.0 / scale
    kept = [
        (cx * inverse_scale, cy * inverse_scale, r * inverse_scale, ratio)
        for cx, cy, r, ratio in kept
    ]

    return kept, mask


def runPipeline(image, llrobot):
    # This is the function a Limelight camera actually calls, once per video
    # frame, when this file is loaded as a "snapscript." Limelight expects it
    # to have exactly this shape: take in an image (and a scratch array called
    # llrobot, which we don't use here) and hand back three things:
    #   1. largestContour - an outline (a list of corner points) marking the
    #      single most important detected object, which Limelight can use for
    #      some of its built-in targeting math.
    #   2. image - the same image, optionally drawn on, which shows up on the
    #      Limelight's web dashboard so a human can see what the camera sees.
    #   3. llpython - a small array of plain numbers that gets sent back to
    #      the robot code, so the robot's program can react to what the camera
    #      found (e.g. "there are 3 balls, and the closest one is over here").

    balls, mask = _detect_balls(image)

    # Default values to return if no pollen balls were found at all: an empty
    # contour, and an llpython array that's all zeros (so robot code can check
    # "is llpython[0] > 0?" to know whether anything was detected).
    largestContour = np.array([])
    llpython = [0.0] * 9

    if balls:
        # balls[0] is the biggest detected ball (the list was sorted by size
        # in _detect_balls), which we treat as the "main" target.
        cx, cy, r, ratio = balls[0]

        # Build a simple square bounding box around that ball's circle and
        # hand it back as the "largestContour." A contour is normally a list
        # of points that trace an object's outline; here we approximate the
        # ball with the four corners of its bounding square, which is a good
        # enough stand-in for Limelight's targeting calculations. Note the
        # unusual nested-list shape ([[x, y]] instead of [x, y]) — that's just
        # the exact array shape OpenCV expects a contour to be in.
        largestContour = np.array(
            [[[int(cx - r), int(cy - r)]], [[int(cx + r), int(cy - r)]],
             [[int(cx + r), int(cy + r)]], [[int(cx - r), int(cy + r)]]]
        )

        # Fill in the numbers we're sending back to the robot: how many balls
        # were found in total, plus the biggest one's x position, y position,
        # and radius (all in pixel coordinates). Slots 4-8 are left as 0 —
        # they're just reserved space in case a future version wants to
        # report more balls, or more details, without changing the array size.
        llpython[0] = float(len(balls))
        llpython[1] = cx
        llpython[2] = cy
        llpython[3] = r

    # Draw a visible marker on EVERY detected ball (not just the biggest one),
    # so a human looking at the Limelight dashboard can see all of them: a
    # thick red circle around the ball's outline, plus a small solid blue dot
    # right at its center point.
    for cx, cy, r, ratio in balls:
        cv2.circle(image, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 255), 3)
        cv2.circle(image, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), -1)

    return largestContour, image, llpython


if __name__ == "__main__":
    # This block only runs when you execute this file directly from the
    # command line (like `python pollenPipeline.py somepicture.jpg`) — it does
    # NOT run when a Limelight camera imports this file and calls
    # runPipeline() itself. It exists purely so a human can quickly test the
    # detector on a single saved photo and see the result pop up on screen,
    # without needing an actual Limelight camera hooked up.

    if len(sys.argv) < 2:
        print("Usage: python pollenPipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    largestContour, outImage, llpython = runPipeline(image, [])
    print("llpython:", llpython)

    cv2.imshow("Pollen Detection", outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
