ذDataset description
Each dataset consists of 2 views taken under several different illuminations and exposures. The files are organized as follows:

SCENE-{perfect,imperfect}/     -- each scene comes with perfect and imperfect calibration (see paper)
  ambient/                     -- directory of all input views under ambient lighting
    L{1,2,...}/                -- different lighting conditions
      im0e{0,1,2,...}.png      -- left view under different exposures
      im1e{0,1,2,...}.png      -- right view under different exposures
  calib.txt                    -- calibration information
  im{0,1}.png                  -- default left and right view
  im1E.png                     -- default right view under different exposure
  im1L.png                     -- default right view with different lighting
  disp{0,1}.pfm                -- left and right GT disparities
  disp{0,1}-n.png              -- left and right GT number of samples (* perfect only)
  disp{0,1}-sd.pfm             -- left and right GT sample standard deviations (* perfect only)
  disp{0,1}y.pfm               -- left and right GT y-disparities (* imperfect only)

Zip files containing the above files (except for the "ambient" subdirectories) for each scene can be downloaded here.

Calibration file format
Here is a sample calib.txt file for one of the full-size training image pairs:

    cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
    cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
    doffs=131.111
    baseline=193.001
    width=2964
    height=1988
    ndisp=280
    isint=0
    vmin=31
    vmax=257
    dyavg=0.918
    dymax=1.516

The calibration files provided with the test image pairs used in the stereo evaluation only contain the first 7 lines, up to the "ndisp" parameter.

Explanation:

    cam0,1:        camera matrices for the rectified views, in the form [f 0 cx; 0 f cy; 0 0 1], where
      f:           focal length in pixels
      cx, cy:      principal point  (note that cx differs between view 0 and 1)

    doffs:         x-difference of principal points, doffs = cx1 - cx0

    baseline:      camera baseline in mm

    width, height: image size

    ndisp:         a conservative bound on the number of disparity levels;
                   the stereo algorithm MAY utilize this bound and search from d = 0 .. ndisp-1

    isint:         whether the GT disparites only have integer precision (true for the older datasets;
                   in this case submitted floating-point disparities are rounded to ints before evaluating)

    vmin, vmax:    a tight bound on minimum and maximum disparities, used for color visualization;
                   the stereo algorithm MAY NOT utilize this information

    dyavg, dymax:  average and maximum absolute y-disparities, providing an indication of
                   the calibration error present in the imperfect datasets.

To convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm] the following equation can be used:

    Z = baseline * f / (d + doffs) 

Note that the image viewer "sv" and mesh viewer "plyv" provided by our software cvkit can read the calib.txt files and provide this conversion automatically when viewing .pfm disparity maps as 3D meshes.
