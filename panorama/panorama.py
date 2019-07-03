import pickle

import numpy as np
import argparse
import imageio
import logging
import sys
from scipy.ndimage.filters import convolve

# Displacements are by default saved to a file after every run. Once you have confirmed your
# LK code is working, you can load saved displacements to save time testing the
# rest of the project.
DEFAULT_DISPLACEMENTS_FILE = "final_displacements.pkl"


def build_A(pts1, pts2):
    """
    Constructs the intermediate matrix A used in the total least squares
    computation of an homography mapping pts1 to pts2.

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 2Nx9 matrix A that we'll use to solve for h
    """

    if pts1.shape != pts2.shape:
        raise ValueError('The source points for homography computation must have the same shape (%s vs %s)' % (
            str(pts1.shape), str(pts2.shape)))
    if pts1.shape[0] < 4:
        raise ValueError('There must be at least 4 pairs of correspondences.')
    num_pts = pts1.shape[0]

    # TODO: Create A which is 2N by 9...
    A = np.zeros(shape=(2 * num_pts, 9))

    # TODO: iterate over the points and populate the rows of A.
    for i in range(num_pts):
        A[2 * i] = [pts1[i][0], pts1[i][1], 1, 0, 0, 0, -pts2[i][0] * pts1[i][0], -pts2[i][0] * pts1[i][1], -pts2[i][0]]
        A[2 * i + 1] = [0, 0, 0, pts1[i][0], pts1[i][1], 1, -pts2[i][1] * pts1[i][0], -pts2[i][1] * pts1[i][1],
                        -pts2[i][1]]
    return A


def compute_H(pts1, pts2):
    """
    Computes an homography mapping one set of co-planar points (pts1)
    to another (pts2).

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 3x3 homography matrix that maps homogeneous coordinates of pts 1 to those in pts2.
    """

    # TODO: Construct the intermediate A matrix using build_A
    A = build_A(pts1, pts2)
    # TODO: Compute the symmetric matrix AtA.
    AtA = np.transpose(A).dot(A)
    # TODO: Compute the eigenvalues and eigenvectors of AtA.
    eig_vals, eig_vecs = np.linalg.eig(AtA)
    # TODO: Determine which eigenvalue is the smallest
    min_eig_val_index = np.argmin(eig_vals)
    # TODO: Return the eigenvector corresponding to the smallest eigenvalue, reshaped as a 3x3 matrix.
    min_eig_vec = eig_vecs.T[min_eig_val_index].reshape(3, 3)
    return min_eig_vec


def bilinear_interp_homography(image, point):
    """
    Looks up the pixel values in an image at a given point using bilinear
    interpolation. point is in the format (x, y).

    Args:
        image:      The image to sample
        point:      A tuple of floating point (x, y) values.

    Returns:
        A 3-dimensional numpy array representing the pixel value interpolated by "point".
    """

    # TODO: extract x and y from point
    x, y = point
    # TODO: Compute i,j as the integer parts of x, y
    i, j = int(x), int(y)
    # TODO: check that i + 1 and j + 1 are within range of the image. if not, just return the pixel at i, j
    if i + 1 >= image.shape[1] or j + 1 >= image.shape[0]:
        return image[j][i]
    # TODO: Compute a and b as the floating point parts of x, y
    a, b = x % 1, y % 1
    # TODO: Take a linear combination of the four points weighted according to the inverse area around them
    # (i.e., the formula for bilinear interpolation)
    enlarged_image = (1 - a) * (1 - b) * image[j][i] + \
                     a * (1 - b) * image[j][i + 1] + \
                     a * b * image[j + 1][i + 1] + \
                     (1 - a) * b * image[j + 1][i]
    return enlarged_image


def apply_homography(H, points):
    """
    Applies the homography matrix H to the provided cartesian points and returns the results
    as cartesian coordinates.

    Args:
        H:      A 3x3 floating point homography matrix.
        points: An Nx2 matrix of x,y points to apply the homography to.

    Returns:
        An Nx2 matrix of points that are the result of applying H to points.
    """

    # TODO: First, transform the points to homogenous coordinates by adding a `1`
    homogenous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # TODO: Apply the homography
    applied_homography = np.array([H.dot(point) for point in homogenous_points])
    # TODO: Convert the result back to cartesian coordinates and return the results
    result = np.array([point[0:2] / point[2] for point in applied_homography])
    return result


def warp_homography(source, target_shape, Hinv):
    """
    Warp the source image into the target coordinate frame using a provided
    inverse homography transformation.

    Args:
        source:         A 3-channel image represented as a numpy array.
        target_shape:   A 3-tuple indicating the desired results height, width, and channels, respectively
        Hinv:           A homography that maps locations in the result to locations in the source image.

    Returns:
        An image of target_shape with source's type containing the source image warped by the homography.
    """

    # TODO: allocation a numpy array of zeros that is size target_shape and the same type as source.
    result = np.zeros(target_shape, dtype=source.dtype)
    # TODO: Iterate over all pixels in the target image
    height, width, _ = target_shape
    for x in range(width):
        for y in range(height):
            # TODO: apply the homography to the x,y location
            source_x, source_y = apply_homography(Hinv, np.array((x, y)).reshape(1, 2))[0]
            # TODO: check if the homography result is outside the source image. If so, move on to next pixel.
            if source_x < 0 or source_x >= source.shape[1] or source_y < 0 or source_y >= source.shape[0]:
                continue
            # TODO: Otherwise, set the pixel at this location to the bilinear interpolation result.
            result[y, x] = bilinear_interp_homography(source, (source_x, source_y))
    # return the output image
    return result


def rectify_image(image, source_points, target_points, crop):
    """
    Warps the input image source_points to the plane defined by target_points.

    Args:
        image:          The input image to warp.
        source_points:  The coordinates in the input image to warp from.
        target_points:  The coordinates to warp the corresponding source points to.
        crop:           If False, all pixels from the input image are shown. If true, the image is cropped to
                        not show any black pixels.
    Returns:
        A new image containing the input image rectified to target_points.
    """

    # TODO: Compute the rectifying homography H that warps the source points to the target points.
    H = compute_H(source_points, target_points)
    # TODO: Apply the homography to a rectangle of the bounding box of the of the image to find the
    # warped bounding box in the rectified space.
    height, width, channels = image.shape
    original_bounding_box = np.array([[0, 0],
                                      [width, 0],
                                      [width, height],
                                      [0, height]])
    warped_bounding_box = apply_homography(H, original_bounding_box)
    # Find the min_x and min_y values in the warped space to keep.
    if crop:
        # TODO: pick the second smallest values of x and y in the warped bounding box
        min_x, min_y = (np.partition(warped_bounding_box[:, 0], 1)[1], np.partition(warped_bounding_box[:, 1], 1)[1])
    else:
        # TODO: Compute the min x and min y of the warped bounding box
        min_x, min_y = np.amin(warped_bounding_box, axis=0)
    # TODO: Compute a translation matrix T such that min_x and min_y will go to zero
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]])
    # TODO: Compute the rectified bounding box by applying the translation matrix to the warped bounding box.
    rectified_bounding_box = apply_homography(T, warped_bounding_box)
    # TODO: Compute the inverse homography that maps the rectified bounding box to the original bounding box
    Hinv = compute_H(rectified_bounding_box, original_bounding_box)
    # Determine the shape of the output image
    if crop:
        # TODO: Determine the second highest X and Y values of the rectified bounding box
        max_x, max_y = (np.partition(rectified_bounding_box[:, 0], -2)[-2], np.partition(rectified_bounding_box[:, 1], -2)[-2])
    else:
        # TODO: Determine the side of the final output image as the maximum X and Y values of the rectified bounding box
        max_x, max_y = np.amax(rectified_bounding_box, axis=0)
    # TODO: Finally call warp_homography to rectify the image and return the result
    result = warp_homography(image, (int(max_y), int(max_x), channels), Hinv)
    return result


def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize//2), ksize//2, ksize)
                    ** 2 / 2) / np.sqrt(2*np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""

    # Cylindrical warping introduces black pixels which should be ignored, and
    # motion in dark regions is difficult to estimate. Generate a binary mask
    # indicating pixels that are valid (average color value > 0.25) in both H
    # and I.
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]
    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    # In other words, compute I_y, I_x, and I_t
    # To achieve this, use a _normalized_ 3x3 sobel kernel and the convolve_img
    # function above. NOTE: since you're convolving the kernel, you need to
    # multiply it by -1 to get the proper direction.
    G_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]) * .125
    G_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]) * .125
    I_t = H - I
    I_x = convolve_img(I, G_x)
    I_y = convolve_img(I, G_y)
    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form AtA. Apply the mask to each product.
    I_xx = I_x * I_x; I_xy = I_x * I_y; I_yy = I_y * I_y; I_xt = I_x * I_t ; I_yt = I_y * I_t
    I_xx = mask * I_xx; I_xy = mask * I_xy; I_yy = mask * I_yy; I_xt = mask * I_xt; I_yt = mask * I_yt
    # Build the AtA matrix and Atb vector. You can use the .sum() function on numpy arrays to help.
    AtA = np.array([[I_xx.sum(), I_xy.sum()],
                    [I_xy.sum(), I_yy.sum()]])
    Atb = np.array([I_xt.sum(),
                    I_yt.sum()]) * -1
    # Solve for the displacement using linalg.solve
    displacement = np.linalg.solve(AtA, Atb).flatten()
    # return the displacement and some intermediate data for unit testing..
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        # Translate the H image by the current displacement (using the translate function above)
        H_trans = translate(H, disp)
        # run Lucas Kanade and update the displacement estimate
        current_disp, _, _ = lucas_kanade(H_trans, I)
        disp += current_disp
    # Return the final displacement
    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Returns:
        An array of images where each image is a blurred and shruken version of the first.
    """

    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.
    kernel = gaussian_kernel()
    # Add image to the the list as the first level
    pyr = [image]
    for level in range(1, levels):
        # Convolve the previous image with the gaussian kernel
        convolved_image = convolve_img(pyr[level - 1], kernel)
        # decimate the convolved image by downsampling the pixels in both dimensions.
        # Note: you can use numpy advanced indexing for this (i.e., ::2)
        convolved_image = convolved_image[::2, ::2]
        # add the sampled image to the list
        pyr.append(convolved_image)
    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float32)

    # Build Gaussian pyramids for the two images.
    pyr1 = gaussian_pyramid(H, levels)
    pyr2 = gaussian_pyramid(I, levels)

    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.
    disp = initial_d / 2.**(levels)
    for level in reversed(range(levels)):
        # Get the two images for this pyramid level.
        img1 = pyr1[level]
        img2 = pyr2[level]
        # Scale the previous level's displacement and apply it to one of the
        # images via translation.
        disp = disp*2
        img1 = translate(img1, disp)
        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        pyr_disp = iterative_lucas_kanade(img1, img2, steps)
        # Update the displacement based on the one you just computed.
        disp += pyr_disp
    # Return the final displacement.
    return disp


def build_panorama(images, shape, displacements, initial_position, blend_width=16):
    # Allocate an empty floating-point image with space to store the panorama
    # with the given shape.
    image_height, image_width = images[0].shape[:2]
    pano_height, pano_width = shape
    panorama = np.zeros((pano_height, pano_width, 3), np.float32)

    # Place the last image, warped to align with the first, at its proper place
    # to initialize the panorama.
    cur_pos = initial_position
    cp = np.round(cur_pos).astype(np.int32)
    panorama[cp[0]: cp[0] + image_height, cp[1]: cp[1] +
             image_width] = translate(images[-1], displacements[-1])

    # Place the images at their final positions inside the panorama, blending
    # each image with the panorama in progress. Use a blending window with the
    # given width.
    for i in range(len(images)):
        cp = np.round(cur_pos).astype(np.int32)

        overlap = image_width - abs(displacements[i][0])
        blend_start = int(overlap / 2 - blend_width / 2)
        blend_start_pano = int(cp[1] + blend_start)

        pano_region = panorama[cp[0]: cp[0] + image_height,
                               blend_start_pano: blend_start_pano+blend_width]
        new_region = images[i][:, blend_start: blend_start+blend_width]

        mask = np.zeros((image_height, blend_width, 1), np.float32)
        mask[:] = np.linspace(0, 1, blend_width)[np.newaxis, :, np.newaxis]
        mask[np.all(new_region == 0, axis=2)] = 0
        mask[np.all(pano_region == 0, axis=2)] = 1

        blended_region = mask * new_region + (1-mask) * pano_region

        blended = images[i].copy("C")
        blended[:, blend_start: blend_start+blend_width] = blended_region
        blended[:, :blend_start] = panorama[cp[0] : cp[0] + image_height, cp[1]: blend_start_pano]

        panorama[cp[0]: cp[0] + blended.shape[0],
                 cp[1]: cp[1] + blended.shape[1]] = blended
        cur_pos += -displacements[i][::-1]
        print("Placed %d." % i)

    # Return the finished panorama.
    return panorama


def mosaic(images, initial_displacements, load_displacements_from):
    """Given a list of N images taken in clockwise order and corresponding
    initial X/Y displacements of shape (N,2), refine the displacements and
    build a mosaic.

    initial_displacement[i] gives the translation that should be appiled to
    images[i] to align it with images[(i+1) % N]."""
    N = len(images)

    if load_displacements_from:
        print("Loading saved displacements...")
        final_displacements = pickle.load(open(load_displacements_from, "rb"))
    else:
        print("Refining displacements with Pyramid Iterative Lucas Kanade...")
        final_displacements = []
        for i in range(N):
            # TODO Use Pyramid Iterative Lucas Kanade to compute displacements from
            # each image to the image that follows it, wrapping back around at
            # the end. A suggested number of levels and steps is 4 and 5
            # respectively. Make sure to append the displacement to
            # final_displacements so it gets saved to disk if desired.
            final_displacements.append(pyramid_lucas_kanade(images[i], images[(i+1) % N], initial_displacements[i], 4, 5))
            # Some debugging output to help diagnose errors.
            print("Image %d:" % i,
                  initial_displacements[i], "->", final_displacements[i], "  ",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -initial_displacements[i]))).mean(), "->",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -final_displacements[i]))).mean()
                  )
        print('Saving displacements to ' + DEFAULT_DISPLACEMENTS_FILE)
        pickle.dump(final_displacements, open(DEFAULT_DISPLACEMENTS_FILE, "wb"))

    # Use the final displacements and the images' shape compute the full
    # panorama shape and the starting position for the first panorama image.
    pano_width = int(np.ceil(images[0].shape[1] + sum([np.abs(i[0]) for i in final_displacements[:-1]])))
    pano_height = int(np.ceil(images[0].shape[0] + sum([i[1] for i in final_displacements[:-1]])))
    initial_pos = [sum([i[1] for i in final_displacements[:-1]]), 0]
    # Build the panorama.
    print("Building panorama...")
    panorama = build_panorama(images, (pano_height, pano_width), final_displacements, initial_pos.copy())
    return panorama, final_displacements


def warp_panorama(images, panorama, final_displacements):
    # Extra credit: Implement this function!
    print("Warping drift...")
    # Resample the panorama image using a linear warp to distribute any vertical
    # drift to all of the sampling points. The final height of the panorama should
    # be equal to the height of one of the images.
    pano_height, pano_width, _ = panorama.shape
    image_height, image_width, _ = images[0].shape

    source_points = np.array([
        0, pano_height,
        0, int(np.ceil(sum([i[1] for i in final_displacements[:-1]]))),
        pano_width, 0,
        pano_width, pano_height - int(np.ceil(sum([i[1] for i in final_displacements[:-1]])))
    ]).reshape(4, 2)

    target_points = np.array([
        0, image_height,
        0, 0,
        pano_width, 0,
        pano_width, image_height
    ]).reshape(4, 2)

    panorama = rectify_image(panorama, source_points, target_points, True)
    # Crop the panorama horizontally so that the left and right edges of the
    # panorama match (making it form a loop).
    panorama = np.delete(
        panorama,
        np.s_[(pano_width - (images[-1].shape[1] - int(np.abs(final_displacements[-1][0])))):pano_width],
        axis=1)
    # Return your final corrected panorama.
    warped = panorama
    return warped


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Creates a mosaic by stitching together a provided set of images.')
    parser.add_argument(
        'input', type=str, help='A txt file containing the images and initial displacement positions.')
    parser.add_argument('output', type=str,
                        help='What image file to save the panorama to.')
    parser.add_argument('--displacements', type=str,
                        help='Load displacements from this pickle file (useful for build_panorama).', default=None)
    args = parser.parse_args()

    filenames, xinit, yinit = zip(
        *[l.strip().split() for l in open(args.input).readlines()])
    xinit = np.array([float(x) for x in xinit])[:, np.newaxis]
    yinit = np.array([float(y) for y in yinit])[:, np.newaxis]
    disps = np.hstack([xinit, yinit])

    images = [imageio.imread(fn)[:, :, :3].astype(
        np.float32)/255. for fn in filenames]

    panorama, final_displacements = mosaic(images, disps, args.displacements)

    result = warp_panorama(images, panorama, final_displacements)
    imageio.imwrite(args.output, result)
