import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import signal as sig

def describe_point(im: np.ndarray, pos: list) -> dict:
    """Crée un descripteur de caractéristique pour un point l'image
    Parameters
    ----------
    im: ndarray
        Image source
    pos: (2,) list
        Position (r,c) dans l'image qu'on souhaite décrire
    Returns
    -------
    d: dict
        Descripteur pour cet indice.
    """
    r = 2  # Rayon du voisinage

    # Descripteur
    d = dict()
    d["pos"] = pos
    d["n"] = (2*r + 1)**2*im.shape[2]  # Nombre de valeurs dans le descripteur
    d["data"] = np.zeros((d["n"],), dtype=float)

    # Valeur du pixel central
    cval = im[pos[0], pos[1], :]

    # Limite du voisinage
    r0 = pos[0] - r if pos[0] - r > 0 else 0
    r1 = pos[0] + r + 1 if pos[0] + r + 1 < im.shape[0] else im.shape[0]-1
    c0 = pos[1] - r if pos[1] - r > 0 else 0
    c1 = pos[1] + r + 1 if pos[1] + r + 1 < im.shape[1] else im.shape[1]-1

    # Extraction et normalisation des valeurs
    values = (im[r0:r1, c0:c1, :].astype(float) - cval).ravel()

    # Intégration dans le descripteur
    d['data'][0:len(values)] = values

    return d


def mark_spot(im: np.ndarray, p: list, color: list = [255, 0, 255]) -> np.ndarray:
    """ Marque la position d'un point dans l'image.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    p: (2,) list
        Position (r,c) du point
    color: (3,) list
        Couleur de la marque
    Returns
    -------
    im: ndarray
        Image marquée.
    """
    r = p[0]
    c = p[1]

    for i in range(-9, 10):
        if r+i < 0 or r+i >= im.shape[0] or c+i < 0 or c+i >= im.shape[1]:
            continue  # ce pixel est à l'extérieur de l'image
        im[r+i, c, 0] = color[0]
        im[r+i, c, 1] = color[1]
        im[r+i, c, 2] = color[2]
        im[r, c+i, 0] = color[0]
        im[r, c+i, 1] = color[1]
        im[r, c+i, 2] = color[2]

    return im


def mark_corners(im: np.ndarray, d: list, n: int) -> np.ndarray:
    """ Marks corners denoted by an array of descriptors.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    d: list
        Coins dans l'image
    n: int
        Nombre de descripteurs à marquer
    Returns
    -------
    im: ndarray
        Image marquée
    """
    m = np.copy(im)
    for i in range(n):
        m = mark_spot(m, d[i]['pos'])
    return m


def smooth_image(im: np.ndarray, sigma: float) -> np.ndarray:
    """Lissage d'une image avec un filtre gaussien.
    Parameters
    ----------
    im: ndarray
        Image à traiter
    sigma: float
        Écart-type pour la gaussienne.
    Returns
    -------
    s: ndarray
        Image lissée
    """
    s = gaussian_filter(im, sigma)
    return s


def structure_matrix(im: np.ndarray, sigma: float) -> np.ndarray:
    """Calcul du tenseur de structure d'un image.
    Parameters
    ----------
    im: ndarray
        Image à traiter (tons de gris et normalisée entre 0 et 1).
    sigma: float
        Écart-type pour la somme pondérée
    Returns
    -------
    S: ndarray
        Tenseur de structure. 1er canal est Ix^2, 2e canal est Iy^2
        le 3e canal est IxIy
    """
    S = np.zeros((*im.shape, 3))

    # filtre de Sobel vertical et horzontal
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # x and y image derivate
    dx = sig.convolve2d(im, sobel_x, mode='same', boundary='symm')
    dy = sig.convolve2d(im, sobel_y, mode='same', boundary='symm')

    # derivees du second ordre (Ix^2, Iy^2, IxIy).
    Ixx = dx*dx
    Ixy = dx*dy
    Iyy = dy*dy

    # construction de la matrice de Structure S (Ixx, Iyy, Ixy)
    S[..., 0] = Ixx
    S[..., 1] = Iyy
    S[..., 2] = Ixy
    S = smooth_image(S, sigma)

    return S


def cornerness_response(S: np.ndarray) -> np.ndarray:
    """Estimation du cornerness de chaque pixel avec le tenseur de structure S.
    Parameters
    ----------
    S: ndarray
        Tenseur de structure de l'image
    Returns
    -------
    R: ndarray
        Une carte de réponse de la cornerness
    """
    R = np.zeros(S.shape[0:2])

    # On utilise la formulation det(S) - alpha * trace(S)^2, alpha = 0.06

    det = S[..., 0] * S[..., 1] - S[..., 2]**2
    trace = S[..., 0] + S[..., 1]
    R = det - 0.06 * (trace**2)

    return R


def nms_image(im: np.ndarray, w: int) -> np.ndarray:
    """Effectue la supression des non-maximum sur l'image des feature responses.
    Parameters
    ----------
    im: ndarray
        Image 1 canal des réponses de caractéristiques (feature response)
    w: int
        Distance à inspecter pour une grande réponse
    Returns
    -------
    r: ndarray
        Image contenant seulement les maximums locaux pour un voisinage de w pixels.
    """
    r = np.copy(im)

    # Pour chaque pixel dans l'image:
    #     Pour chaque voisin dans w:
    #         Si la réponse du voisin est plus grande que la réponse du pixel:
    #             Assigner à ce pixel une très petite réponse (ex: -np.inf)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(-w, w):
                for l in range(-w, w):
                    # check if the neighbor is in the image and if the neighbor is bigger than the pixel.
                    try:
                        # if the neighbor is bigger than the pixel, assign a very small response.
                        if r[i, j] < r[i+k, j+l]:
                            r[i, j] = -np.inf
                    except IndexError:
                        pass

    return r


def harris_corner_detector(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray:
    """ Détecteur de coin de Harris, et extraction des caractéristiques autour des coins.
    Parameters
    ----------
    im: ndarray
        Image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    -------
    d: list
        Liste des descripteurs pour chaque coin dans l'image
    """
    img = im.mean(axis=2)  # Convert to grayscale
    img = (img.astype(float) - img.min()) / (img.max() - img.min())

    # Calculate structure matrix
    S = structure_matrix(img, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)


    count = 0  # changez ceci
    point = []
    # checking if the response is above the threshold and if it is, add it to the list of points. and increment the count.
    for i in range(Rnms.shape[0]):
        for j in range(Rnms.shape[1]):
            if Rnms[i, j] > thresh:
                count += 1
                point.append((i, j))

    # printing the number of points found.
    print("Nombre de coins:", count)

    n = count  # <- fixer n = nombre de coins dans l'image
    d = []

    for i in range(n):
        d.append(describe_point(im, point[i]))
    return d


def detect_and_draw_corners(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray:
    """ Trouve et dessine les coins d'une image
    Parameters
    ----------
    im: ndarray
        L'image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness.
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    m: ndarray
        Image marqué avec les coins détectés
    """
    d = harris_corner_detector(im, sigma, thresh, nms)
    m = mark_corners(im, d, len(d))
    return m

