from harris_image import harris_corner_detector, mark_corners
import math
import numpy as np
import random
from functools import cmp_to_key


def make_translation_homography(dr: float, dc: float) -> np.ndarray:
    """Create a translation homography
    Parameters
    ----------
    dr: float
        Translation along the row axis
    dc: float
        Translation along the column axis
    Returns
    -------
    H: np.ndarray
        Homography as a 3x3 matrix
    """
    H = np.zeros((3, 3))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[0, 2] = dr  # Row translation
    H[1, 2] = dc  # Col translation


def match_compare(a: float, b: float) -> int:
    """ Comparator for matches
    Parameters
    ----------
    a,b : float
        distance for each match to compare.
    Returns
    -------
    result of comparison, 0 if same, 1 if a > b, -1 if a < b.
    """
    comparison = 0
    if a < b:
        comparison = -1
    elif a > b:
        comparison = 1
    else:
        comparison = 0
    return comparison


def both_images(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Place two images side by side on canvas, for drawing matching pixels.
    Parameters
    ----------
    a,b: ndarray
        Images to place
    Returns
    -------
    c: ndarray
        image with both a and b side-by-side.
    """
    width = a.shape[1] + b.shape[1]
    height = a.shape[0] if a.shape[0] > b.shape[0] else b.shape[0]
    channel = a.shape[2] if a.shape[2] > b.shape[2] else b.shape[2]

    both = np.zeros((height, width, channel), dtype=a.dtype)
    both[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    both[0:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1], 0:b.shape[2]] = b

    return both


def draw_matches(a: np.ndarray, b: np.ndarray, matches: list, inliers: int) -> np.ndarray:
    """Draws lines between matching pixels in two images.
    Parameters
    ----------
    a, b: ndarray
        two images that have matches.
    matches: list
        array of matches between a and b.
    inliers: int
        number of inliers at beginning of matches, drawn in green.
    Returns
    -------
    c: ndarray
        image with matches drawn between a and b on same canvas.
    """
    both = both_images(a, b)
    n = len(matches)
    for i in range(n):
        r1 = matches[i]['p'][0]  # Coordonnée y du point p
        r2 = matches[i]['q'][0]  # Coordonnée y du point q
        c1 = matches[i]['p'][1]  # Coordonnée x du point p
        c2 = matches[i]['q'][1]  # Coordonnée x du point q
        for c in range(c1, c2 + a.shape[1]):
            r = int((c-c1)/(c2 + a.shape[1] - c1)*(r2 - r1) + r1)
            both[r, c, 0] = (0 if i < inliers else 255)
            both[r, c, 1] = (255 if i < inliers else 0)
            both[r, c, 2] = 0
    return both


def draw_inliers(a: np.ndarray, b: np.ndarray, H: np.ndarray, matches: list, thresh: float) -> np.ndarray:
    """ Draw the matches with inliers in green between two images.
    Parameters
    ----------
    a, b: ndarray
        two images to match.
    H: ndarray
        Homography matrix
    matches: list
        array of matches between a and b
    thresh: float
        Threshold to define inliers
    Returns
    -------
    lines: ndarray
        Modified images with inliers
    """
    n_inliers, new_matches = model_inliers(H, matches, thresh)
    lines = draw_matches(a, b, new_matches, n_inliers)
    return lines


def find_and_draw_matches(a: np.ndarray, b: np.ndarray, sigma: float = 2, thresh: float = 3, nms: int = 3) -> np.ndarray:
    """ Find corners, match them, and draw them between two images.
    Parameters
    ----------
    a, b: np.ndarray
         images to match.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 1-5
    nms: int
        window to perform nms on. Typical: 3
    Returns
    -------
    lines: np.ndarray
        Images with inliers
    """
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)
    m = match_descriptors(ad, bd)
    H = RANSAC(m, 10, 10000, 15)
    a = mark_corners(a, ad, len(ad))
    b = mark_corners(b, bd, len(bd))
    lines = draw_inliers(a, b, H, m, 20)

    return lines


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates L1 distance between to floating point arrays.
    Parameters
    ----------
    a, b: list or np.ndarray
        arrays to compare.
    Returns
    -------
    l1: float
        l1 distance between arrays (sum of absolute differences).
    """
    l1 = 0

    # calcul de la distance l1
    for i in range(len(a)):
        l1 += abs(a[i]-b[i])

    return l1


def match_descriptors(a: list, b: list) -> list:
    """Finds best matches between descriptors of two images.
    Parameters
    ----------
    a, b: list
        array of descriptors for pixels in two images.
    Returns
    -------
    matches: list
        best matches found. each descriptor in a should match with at most
        one other descriptor in b.
    """
    an = len(a)
    bn = len(b)
    matches = []

    # record ai as the index in a and bi as the index in b.
    for j in range(an):

        # record ai as the index in a and bi as the index in b.

        minimum_distance = l1_distance(a[j]["data"], b[0]["data"])
        bind = 0  # <- find the best match
        # finding the best discriptro in b for a[j]

        # finding the best match for a[j] in b and storing it minimum_distance.
        for i in range(1, bn):
            distance = l1_distance(a[j]["data"], b[i]["data"])
            if distance < minimum_distance:
                bind = i
                minimum_distance = distance

        matches.append({})
        matches[j]['ai'] = j
        matches[j]['bi'] = bind  # <- should be index in b.
        matches[j]['p'] = a[j]['pos']
        matches[j]['q'] = b[bind]['pos']
        # <- should be the smallest L1 distance!
        matches[j]['distance'] = minimum_distance

    seen = []
    filtered_matches = []

    # Sort matches based on distance using match_compare and sort.
    # Then throw out matches to the same element in b. Use seen to keep track.
    # Each point should only be a part of one match.
    # Some points will not be in a match.
    # In practice just bring good matches to front of list.

    sortedM = sorted(matches, key=cmp_to_key(
        lambda x, y: match_compare(x['distance'], y['distance'])))
    for i in range(len(sortedM)):
        if sortedM[i]['bi'] not in seen:
            filtered_matches.append(sortedM[i])
            seen.append(sortedM[i]['bi'])

    matches = filtered_matches

    return matches


def project_point(H, p):
    """ Apply a projective transformation to a point.
    Parameters
    ----------
    H: np.ndarray
        homography to project point, of shape 3x3
    p: list
        point to project.
    Returns
    -------
    q: list
        point projected using the homography.
    """

    c = np.zeros((3, 1))

    # Remember that homogeneous coordinates are equivalent up to scalar.
    # Have to divide by.... something...
    q = [0, 0]

    c[0][0] = p[0]
    c[1][0] = p[1]
    c[2][0] = 1
    hc = np.matmul(H, c)
    # normalizing
    q[0] = hc[0][0]/hc[2][0]
    q[1] = hc[1][0]/hc[2][0]

    return q


def point_distance(p, q):
    """ Calculate L2 distance between two points.
    Parameters
    ----------
    p, q: list
        points.
    Returns
    -------
    l2: float
        L2 distance between them.
    """
    l2 = 0

    l2 = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

    return l2


def model_inliers(H: np.ndarray, matches: list, thresh: float) -> tuple:
    """Count number of inliers in a set of matches. Should also bring inliers to the front of the array.
    Parameters
    ----------
    H: np.ndarray
        homography between coordinate systems.
    matches: list
        matches to compute inlier/outlier.
    thresh: float
        threshold to be an inlier.
    Returns
    -------
    count: int
        number of inliers whose projected point falls within thresh of their match in the other image.
    matches: list
        Should also rearrange matches so that the inliers are first in the array. For drawing.
    """
    count = 0
    new_matches = []  # To reorder the matches

    # i.e. distance(H*p, q) < thresh
    # Also, sort the matches m so the inliers are the first 'count' elements.
    for i in range(len(matches)):
        p = matches[i]['p']
        q = matches[i]['q']
        distance = point_distance(project_point(H, p), q)
        if distance < thresh:
            # insert at the front of the new_matches list.
            new_matches.insert(0, matches[i])
            count += 1
        else:
            # insert at the end of the new_matches list.
            new_matches.append(matches[i])

    return (count, new_matches)


def randomize_matches(matches: list) -> list:
    """ Randomly shuffle matches for RANSAC.
    Parameters
    ----------
    matches: list
        matches to shuffle in place
    Returns
    -------
    shuffled_matches: list
        Shuffled matches
    """


    for i in range(len(matches)):
        j = random.randint(0, len(matches)-1)
        temp = matches[i]
        matches[i] = matches[j]
        matches[j] = temp

    return matches


def compute_homography(matches: list, n: int) -> np.ndarray:
    """Computes homography between two images given matching pixels.
    Parameters
    ----------
    matches: list
        matching points between images.
    n: int
        number of matches to use in calculating homography.
    Returns
    -------
    H: np.ndarray
        matrix representing homography H that maps image a to image b.
    """
    assert n >= 4, "Underdetermined, use n>=4"

    M = np.zeros((n*2, 8))
    b = np.zeros((n*2, 1))

    for i in range(n):
        r = float(matches[i]['p'][0])
        rp = float(matches[i]['q'][0])
        c = float(matches[i]['p'][1])
        cp = float(matches[i]['q'][1])

        M[2*i][0] = r
        M[2*i][1] = c
        M[2*i][2] = 1
        M[2*i][6] = -r*rp
        M[2*i][7] = -c*rp
        M[2*i+1][3] = r
        M[2*i+1][4] = c
        M[2*i+1][5] = 1
        M[2*i+1][6] = -r*cp
        M[2*i+1][7] = -c*cp
        b[2*i][0] = rp
        b[2*i+1][0] = cp

    # Solve the linear system
    if M.shape[0] == M.shape[1]:
        a = np.linalg.solve(M, b)
    else:  # Over-determined, using least-squared
        a = np.linalg.lstsq(M, b, rcond=None)
        a = a[0]
    # If a solution can't be found, return empty matrix;
    if a is None:
        return None

    H = np.zeros((3, 3))

    H[0][0] = a[0][0]
    H[0][1] = a[1][0]
    H[0][2] = a[2][0]
    H[1][0] = a[3][0]
    H[1][1] = a[4][0]
    H[1][2] = a[5][0]
    H[2][0] = a[6][0]
    H[2][1] = a[7][0]
    H[2][2] = 1

    return H


def RANSAC(matches: list, thresh: float, k: int, cutoff: int):
    """Perform RANdom SAmple Consensus to calculate homography for noisy matches.
    Parameters
    ----------
    matches: list
        set of matches.
    thresh: float
        inlier/outlier distance threshold.
    k: int
        number of iterations to run.
    cutoff: int
        inlier cutoff to exit early.
    Returns
    -------
    Hb: np.ndarray
        matrix representing most common homography between matches.
    """
    best = 0
    Hb = make_translation_homography(0, 256)
    # RANSAC algorithm.
    # for k iterations:
    #     shuffle the matches
    #     compute a homography with a few matches (how many??)
    #     if new homography is better than old (how can you tell?):
    #         compute updated homography using all inliers
    #         remember it and how good it is
    #         if it's better than the cutoff:
    #             return it immediately
    # if we get to the end return the best homography

    print("Initial Hb:", Hb)
    for i in range(k):
        matches = randomize_matches(matches)
        Hi = compute_homography(matches, 4)
        inliers, matchOne = model_inliers(Hi, matches, thresh)
        matches = matchOne
        if (inliers > best):
            print("new inliers :", inliers, ">", best, "plus:", i)
            print("-------------------------------------------")
            Hi2 = compute_homography(matches, inliers)
            best = inliers
            Hb = Hi2
            inliersTwo, matchTwo = model_inliers(Hb, matches, thresh)
            matches = matchTwo
            if inliersTwo >= cutoff:
                best = inliersTwo
                print("Best Homography: ", Hb.shape, "plus", i)
                print("Best Homography: ", Hb, "plus", i)
                print("-------------------------------------------------")
                return Hb
    print("k: ", i)
    print("inliers: ", best)
    print("Hb1: ", Hb)
    return Hb


def combine_images(a, b, H):
    """ Stitches two images together using a projective transformation.
    Parameters
    ----------
    a, b: ndarray
        Images to stitch.
    H: ndarray
        Homography from image a coordinates to image b coordinates.
    Returns
    -------
    c: ndarray
        combined image stitched together.
    """
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    c1 = project_point(Hinv, [0, 0])
    c2 = project_point(Hinv, [b.shape[0], 0])
    c3 = project_point(Hinv, [0, b.shape[1]])
    c4 = project_point(Hinv, [b.shape[0], b.shape[1]])

    # Find top left and bottom right corners of image b warped into image a.
    topleft = [0, 0]
    botright = [0, 0]
    botright[0] = int(max([c1[0], c2[0], c3[0], c4[0]]))
    botright[1] = int(max([c1[1], c2[1], c3[1], c4[1]]))
    topleft[0] = int(min([c1[0], c2[0], c3[0], c4[0]]))
    topleft[1] = int(min([c1[1], c2[1], c3[1], c4[1]]))

    # Find how big our new image should be and the offsets from image a.
    dr = int(min(0, topleft[0]))
    dc = int(min(0, topleft[1]))
    h = int(max(a.shape[0], botright[0]) - dr)
    w = int(max(a.shape[1], botright[1]) - dc)

    # Can disable this if you are making very big panoramas.
    # Usually this means there was an error in calculating H.
    if w > 7000 or h > 7000:
        print("output too big, stopping.")
        return np.copy(a)

    c = np.zeros((h, w, a.shape[2]), dtype=a.dtype)

    # Paste image a into the new image offset by dr and dc.
    # dc is negative and dr is usually equal to zero because our panorama is horizontal
    for k in range(a.shape[2]):
        for j in range(a.shape[1]):
            for i in range(a.shape[0]):
                c[i-dr, j-dc, k] = a[i, j, k]


    # You should loop over some points in the new image (which? all?)
    # and see if their projection from a coordinates to b coordinates falls
    # inside of the bounds of image b. If so, use bilinear interpolation to
    # estimate the value of b at that projection, then fill in image c.

    for k in range(c.shape[2]):
        for j in range(topleft[1], botright[1]):
            for i in range(topleft[0], botright[0]):
                p = interpolation(project_point(H, [i, j])) # projeter le point p dans les coordonnees de b
                if p[0] >= 0 and p[1] >= 0 and p[0] < b.shape[0] and p[1] < b.shape[1]:
                    c[i-dr,j-dc,k] = b[p[0], p[1], k]

    return c


def panorama_image(a, b, sigma=2, thresh=0.0003, nms=3, inlier_thresh=5, iters=10000, cutoff=15):
    """ Create a panoramam between two images.
    Parameters
    ----------
    a, b: ndarray
        images to stitch together.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 0.0001-0.0005
    nms: int
        window to perform nms on. Typical: 3
    inlier_thresh: float
        threshold for RANSAC inliers. Typical: 2-5
    iters: int
        number of RANSAC iterations. Typical: 1,000-50,000
    cutoff: int
        RANSAC inlier cutoff. Typical: 10-100
    """
    # Calculate corners and descriptors
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)

    # Find matches
    m = match_descriptors(ad, bd)

    # Run RANSAC to find the homography
    H = RANSAC(m, inlier_thresh, iters, cutoff)

    # Stitch the images together with the homography
    comb = combine_images(a, b, H)
    return comb


def interpolation(p):
    return [round(p[0]), round(p[1])]
