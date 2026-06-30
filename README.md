# OpenCVDemo

OpenCV demo using examples from FTC.  The `Exercises` directory
contains 10 python programs with various pieces missing.  Your job is
to fill in those missing pieces and learn about OpenCV along the way.

Solutions are in the `Solutions` directory, but don't peek unless you
are really stuck.  Work though it.

You might want to clone our 2024-2025
[LimeLight](https://github.com/GearUp12499-org/INTO-THE-DEEP-Limelight)
repo.  It contains several additional pictures to test the exercise
on.  They are in the `shuban-789/cvlocal/images` subdirectory.  

## Windows

You will need python and git.  Python can be installed from the
Microsoft Store.  This demo works with python 3.13.  A git installer
can be downloaded from the [git
store](https://git-scm.com/install/windows).

Once both are installed, open Windows PowerShell.  You can use notepad
as your editor or download
[notepad++](https://notepad-plus-plus.org/downloads/).

```
PS C:\Users\name> git clone https://github.com/wfdoran/OpenCVDemo
PS C:\Users\name> cd OpenCVDemo\Exercises
PS C:\Users\name\OpenCVDemo\Exercises> notepad exercise1.py

<Edit the file filling in the ???>

PS C:\Users\name\OpenCVDemo\Exercises> python3 exercise1.py
```

For windows, be sure to uncomment
```
from gear_up_utils import win_imshow
```

and switch

```
# cv2.imshow(???, ???)      # Fix ME!
win_imshow(???, ???)    # Windows: Fix ME!
```

Notice the print out to the PowerShell when you click on the image.

Finally, focus on the image window and hit any key to close the program.




