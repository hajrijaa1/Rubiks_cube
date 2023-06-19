import numpy as np

#Kvaternion Rotacija: Predstavljanje 3D rotacije putem Kvaterniona.
class Kvaternion:

    """
        Konstruisanje Kvaterniona od jediničnog vektora v i ugla rotacije theta

        Parametri:
        v - niz jediničnih vektora, vektori moraju biti normalizirani
        theta - niz uglova rotacije koji su predstavljeni u radijanima (shape = v.shape[:-1])

        Funkcija vraća
        q - kvaternion objekat (predstavlja rotacije)
    """
    @classmethod
    def kreiraj_kvaternion_od_v_i_theta(cls, v, theta):
        #predstavljanje theta uglova u obliku niza
        theta = np.asarray(theta)
        #predstavljanje vektora u obliku niza
        v = np.asarray(v)
        #kreiranje nizova sinus vrijednosti i kosinus vrijednosti svih uglobva
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    """
    Inicijalizacija kvaterniona
    """
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)


    """
    Množenje dva kvaterniona (ne množe se skalarno)
    """
    def __mul__(self, other):

        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1] - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0])],
                       dtype=np.float, order='F').T
        return self.__class__(ret.reshape(return_shape))

    """
    kvaternion_kao_v_theta vraća ekvivalent v i theta normaliziranog Kvaterniona
    """
    def kvaternion_kao_v_theta(self):
        x = self.x.reshape((-1, 4)).T

        # Izračunavanje theta vrijednosti
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # Izračunavanje jediničnog vektora
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # Reshape rezultata
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    """
    Funkcija kvaternion_kao_rotacijska_matrica vraća rotacijsku matricu normaliziranog Kvaterniona
    """
    def kvaternion_kao_rotacijska_matrica(self):
        v, theta = self.kvaternion_kao_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    """
    Rotacija kvaterniona (korištenje funkcije kvaternion_kao_rotacijska_matrica)
    """
    def rotate(self, points):
        M = self.kvaternion_kao_rotacijska_matrica()
        return np.dot(points, M.T)

"""
projektne_tacke funkcija koristi Kvaternion q i prikaz v

Parametri:
points - niz posljednje dimenzije 3
q - kvaternion, reprezentacija rotacije
view - vektor dužine-3 koji daje tačku gledišta
vertical - smjer y ose prikaza.

Funkcija vraća:
proj - niz projektnih tačaka, ima isti shape kao i points
"""
def projektne_tacke(points, q, view, vertical=[0, 1, 0]):

    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("Vertikala je paralelna sa v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # Jedinični vektor koji odgovara vertikali
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # Normalizacija lokacije korisnika - predstavlja z osu
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # Rotacija tačaka
    R = q.kvaternion_kao_rotacijska_matrica()
    Rpts = np.dot(points, R.T)

    # Projektovanje tačaka na prikaz
    dpoint = Rpts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans =  list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir),
                     -np.dot(dpoint, zdir)]).transpose(trans)
