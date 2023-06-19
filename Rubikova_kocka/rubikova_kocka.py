import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from projekcija import Kvaternion, projektne_tacke
import time
from multiprocessing import Process

plt.rcParams['toolbar'] = 'None' 
plt.rcParams.update({'figure.max_open_warning': 60})

"""
Svaka strana je predstavljena sa nizom : [v1, v2, v3, v4, v1] (vrhovi strane) [5,3]

Svaki kvadratić je predstavljen nizom: [v1a, v1b, v2a, v2b, v3a, v3b, v4a, v4b, v1a] [9, 3]

U oba slučaja, početna tačka se ponavlja dva puta kako bi se zatvorio poligon (kvadrat)

Svaka strana ima svoj centroid (centralni kvadratić) kojem se pridodaje broj stranice kako bi ispavno vršio lexsort.
Centroid je jednak sum_i[vi].

Boje se dobijaju pomoću indeksa boje i tabele iz koje se čitaju vrijednosti.

Sa svim stranicama u NxNxN kocki, postoje sljedeći nizovi:

  centroidi.shape = (6 * N * N, 4)  - niz centroida 
  stranice.shape = (6 * N * N, 5, 3) - stranice sa 5 vrhova (4 ali se prvi ponavlja dva puta) a svaki ima x, y i z koordinate
  kvadrati.shape = (6 * N * N, 9, 3) - svaka stranica ima 9 kvadratića (kocka 3 x 3) sa x,y i z koordinatom
  boje.shape = (6 * N * N,) - boje 6 boja * Ndimenzija * Ndimenzija (54 za 3 x 3)

Kanonski poredak se pronalazi na sljedeći način: ind = np.lexsort(centroidi.T)
(Nakon bilo koje rotacije, ovo se može koristiti za brzo vraćanje kocke u kanonski položaj)
"""

class Kocka:
    # atributi kocke
    zadana_boja = "#000000"  #crna
    boje_stranica = [  "#ffffff", #bijela
                        "#ffcf00", #žuta
                        "#00008f", #plava
                        "#009f0f", #zelena
                        "#ff6f00", #narandžasta
                        "#cf0000", #crvena
                        "#b6b8b7", #siva
                        "none"]
    osnovna_strana = np.array([
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]], dtype=float)
    sirina_kvadrata = 0.9
    margina_kvadrata = 0.5 * (1. - sirina_kvadrata)
    debljina_kvadrata = 0.001
    (d1, d2, d3) = (1 - margina_kvadrata,  1 - 2 * margina_kvadrata, 1 + debljina_kvadrata)
    osnovni_kvadrat = np.array([
        [d1, d2, d3], 
        [d2, d1, d3],
        [-d2, d1, d3], 
        [-d1, d2, d3],
        [-d1, -d2, d3], 
        [-d2, -d1, d3],
        [d2, -d1, d3], 
        [d1, -d2, d3],
        [d1, d2, d3]], dtype=float)

    osnovni_centroid_stranice = np.array([[0, 0, 1]])
    osnovni_centroid_kvadratica = np.array([[0, 0, 1 + debljina_kvadrata]])

    # Definisanje uglova rotacije i ose za šest strana kocke
    x, y, z = np.eye(3)
    rots = [Kvaternion.kreiraj_kvaternion_od_v_i_theta(np.eye(3)[0], theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Kvaternion.kreiraj_kvaternion_od_v_i_theta(np.eye(3)[1], theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    # Definisanje pokreta
    rjecnik_stranica = dict(
        F = z, 
        B = -z,
        R = x, 
        L = -x,
        U = y, 
        D = -y
        )

    """
    Inicijalizacija kocke
    """
    def __init__(self, N=3):
        self.N = N
        self.boja_plastike = self.zadana_boja
        self.face_boje = self.boje_stranica

        self._move_list = []
        self._inicijalizacija_nizova()

    """
    Inicijalizacija nizova - centroida, stranica i kvadrata
    Kasnije se vrše translacije (pomjeranje svih tačaka figure za istu udaljenost u određenom smjeru) i rotacije (pomjeranje za određeni ugao) na ispravne pozicije
    """
    def _inicijalizacija_nizova(self):

        # Definisanje N² translacija za svaku stranu kocke
        sirina_strane_kocke = 2. / self.N
        translacije = np.array([[[-1 + (i + 0.5) * sirina_strane_kocke,
                                   -1 + (j + 0.5) * sirina_strane_kocke, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Kreiranje nizova za centroide, stranice, kvadratiće i boje
        centroidi_stranica = []
        stranice = []
        centroidi_kvadrata = []
        kvadrati = []
        boje = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            M = self.rots[i].kvaternion_kao_rotacijska_matrica()
            stranice_t = np.dot(factor * self.osnovna_strana + translacije, M.T)
            stranice.append(stranice_t)

            kvadrati_t = np.dot(factor * self.osnovni_kvadrat + translacije, M.T)
            kvadrati.append(kvadrati_t)

            centroidi_kvadrata_t = np.dot(self.osnovni_centroid_kvadratica + translacije, M.T)
            centroidi_kvadrata_t = centroidi_kvadrata_t.reshape((-1, 3))
            centroidi_kvadrata.append(centroidi_kvadrata_t)

            centroidi_stranica_t = np.dot(self.osnovni_centroid_stranice + translacije, M.T)
            boje_i = i + np.zeros(centroidi_stranica_t.shape[0], dtype=int)
            # dodavanje ID stranice centroidu za lex sortiranje
            centroidi_stranica_t = np.hstack([centroidi_stranica_t.reshape(-1, 3), boje_i[:, None]])
            centroidi_stranica.append(centroidi_stranica_t)

            
            boje.append(boje_i)

        self._centroidi_stranica = np.vstack(centroidi_stranica)
        self._stranice = np.vstack(stranice)
        self._centroidi_kvadrata = np.vstack(centroidi_kvadrata)
        self._kvadrati = np.vstack(kvadrati)
        self._boje = np.concatenate(boje)

        self._sortiranje_stranica()

    """
    Sortiranje stranica koristeći lexsort
    """
    def _sortiranje_stranica(self):
        ind = np.lexsort(self._centroidi_stranica.T)
        self._centroidi_stranica = self._centroidi_stranica[ind]
        self._centroidi_kvadrata = self._centroidi_kvadrata[ind]
        self._kvadrati = self._kvadrati[ind]
        self._boje = self._boje[ind]
        self._stranice = self._stranice[ind]

    """
    Miješanje kocke sa n nasumičnih poteza
    """
    def izmijesaj(self):
        f = np.random.choice(['F', 'B', 'R', 'L', 'U', 'D'])
        n = np.random.choice([-1, 1])
        l = np.random.randint(self.N)
        plt.pause(0.01)
        self.rotiraj_stranicu(f, n,  l)
    """
    Rotiranje stranice
    """
    def rotiraj_stranicu(self, f, n=1, layer=0):
        time.sleep(0.1)
        plt.pause(0.01)
        if f == 'U':
            print(" Rotiram prema gore")
        elif f == 'D':
            print(" Rotiram prema dolje")
        elif f == 'L':
            print(" Rotiram ulijevo")
        elif f == 'R':
            print(" Rotiram udesno")
        elif f == 'B':
            print(" Rotiram unazad")
        elif f == 'F':
            print(" Rotiram naprijed")
        if layer < 0 or layer >= self.N:
            raise ValueError('Sloj bi trebao biti između 0 i N-1')

        try:
            f_last, n_last, layer_last = self._move_list[-1]
        except:
            f_last, n_last, layer_last = None, None, None

        if (f == f_last) and (layer == layer_last):
            ntot = (n_last + n) % 4
            if abs(ntot - 4) < abs(ntot):
                ntot = ntot - 4
            if np.allclose(ntot, 0):
                self._move_list = self._move_list[:-1]
            else:
                self._move_list[-1] = (f, ntot, layer)
        else:
            self._move_list.append((f, n, layer))
        
        v = self.rjecnik_stranica[f]
        r = Kvaternion.kreiraj_kvaternion_od_v_i_theta(v, n * np.pi / 2)
        M = r.kvaternion_kao_rotacijska_matrica()

        proj = np.dot(self._centroidi_stranica[:, :3], v)
        sirina_strane_kocke = 2. / self.N
        flag = ((proj > 0.9 - (layer + 1) * sirina_strane_kocke) & (proj < 1.1 - layer * sirina_strane_kocke))

        for x in [self._kvadrati, self._centroidi_kvadrata, self._stranice]:
            x[flag] = np.dot(x[flag], M.T)
        self._centroidi_stranica[flag, :3] = np.dot(self._centroidi_stranica[flag, :3], M.T)

    """
    Iscrtavanje interaktivne kocke
    """
    def nacrtaj_interaktivnu(self):
        fig = plt.figure(figsize=(10, 10))
        fig.add_axes(InteraktivnaKocka(self))
        return fig


class InteraktivnaKocka(plt.Axes):
    def __init__(self, 
                cube=None,
                 view=(0, 0, 10),
                 fig=None, 
                 rect=[0, 0.16, 1, 0.84],
                 **kwargs):
        if cube is None:
            self.cube = Kocka(3)
        elif isinstance(cube, Kocka):
            self.cube = cube
        else:
            self.cube = Kocka(cube)

        self._view = view
        self._start_rot = Kvaternion.kreiraj_kvaternion_od_v_i_theta((1, -1, 0), -np.pi / 6)

        if fig is None:
            fig = plt.gcf()
            fig.canvas.manager.set_window_title('Rješavanje rubikove kocke upotrebom paralelizma')

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # postavljanje defaultnih vrijednosti i iscrtavanje
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-2.0, 2.0)),
                           ylim=kwargs.get('ylim', (-2.0, 2.0)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteraktivnaKocka, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Definisanje pokreta gore/dolje sa strelicama ili pokretom miša
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        #  Definisanje pokreta lijevo/desno sa strelicama ili pokretom miša
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Interne varijablje stanja
        self._active = False  # true kada je miš iznad ose
        self._button1 = False  # true kada se pritisne dugme 1
        self._event_xy = None  # pohraniti xy poziciju događaja miša
        self._digit_flags = np.zeros(10, dtype=bool)  # da li je pritisnuta cifra

        self._current_rot = self._start_rot  #trenutno stanje rotacije
        self._face_polys = None
        self._sticker_polys = None

        self._nacrtaj_kocku()

        self.figure.canvas.mpl_connect('button_press_event', self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event', self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        self.figure.canvas.mpl_connect('key_release_event', self._key_release)

        self._initialize_widgets(self.cube)

        # Ispis instrukcija
        self.figure.text(0.05, 0.05,
                         "Pomjeranje kocke se vrši mišem\n" +
                         "Miješaj kocku koristeći slova:\n" +
                         "U - gore \n" +
                         "D - dolje \n" +
                         "L - lijevo \n" +
                         "R - desno \n" +
                         "B - nazad \n" +
                         "F - naprijed \n",
                         size=10)

    """
    Inicijalizacija - dugme za reset i dugme za rješavanje kocke
    """
    def _initialize_widgets(self, kocka):

        self._ax_randomize = self.figure.add_axes([0.35, 0.05, 0.2, 0.05])
        self._btn_randomize = widgets.Button(self._ax_randomize, 'Izmiješaj kocku')
        self._btn_randomize.on_clicked(self._izmijesaj_kocku)

        self._ax_reset = self.figure.add_axes([0.75, 0.05, 0.2, 0.05])
        self._btn_reset = widgets.Button(self._ax_reset, 'Resetuj prikaz')
        self._btn_reset.on_clicked(self._resetuj_prikaz)

        self._ax_solve = self.figure.add_axes([0.55, 0.05, 0.2, 0.05])
        self._btn_solve = widgets.Button(self._ax_solve, 'Riješi kocku')
        self._btn_solve.on_clicked(self._rijesi_kocku)


    def _projektuj_tacke(self, pts):
        return projektne_tacke(pts, self._current_rot, self._view, [0, 1, 0])

    def _nacrtaj_kocku(self):
        kvadrati = self._projektuj_tacke(self.cube._kvadrati)[:, :, :2]
        stranice = self._projektuj_tacke(self.cube._stranice)[:, :, :2]
        centroidi_stranica = self._projektuj_tacke(self.cube._centroidi_stranica[:, :3])
        centroidi_kvadrata = self._projektuj_tacke(self.cube._centroidi_kvadrata[:, :3])

        boja_plastike = self.cube.boja_plastike
        boje = np.asarray(self.cube.face_boje)[self.cube._boje]
        face_zorders = -centroidi_stranica[:, 2]
        sticker_zorders = -centroidi_kvadrata[:, 2]

        if self._face_polys is None:
            # inicijalizacija: Kreiranje poligona i dodavanje ivica 
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(boje)):
                fp = plt.Polygon(xy = stranice[i], facecolor=boja_plastike, zorder=face_zorders[i])
                sp = plt.Polygon(xy = kvadrati[i], facecolor=boje[i], zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # Ažuriranje poligona
            for i in range(len(boje)):
                self._face_polys[i].set_xy(stranice[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(boja_plastike)

                self._sticker_polys[i].set_xy(kvadrati[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(boje[i])

        self.figure.canvas.draw()

    """
    Rotiranje kocke
    """
    def rotiraj(self, rot):
        self._current_rot = self._current_rot * rot

    """
    Rotiranje stranica
    """
    def rotiraj_stranicu(self, face, turns=1, layer=0, steps=1):
        paralelno_vrijeme = 0
        standardno_vrijeme = 0
        if not np.allclose(turns, 0):
            for i in range(steps):
                start_time = time.perf_counter()
                p = Process(target=self.cube.rotiraj_stranicu, args=(face, turns * 1. / steps, layer))    
                finish_time = time.perf_counter()
                paralelno_vrijeme = finish_time - start_time

                start_time = time.perf_counter()
                self.cube.rotiraj_stranicu(face, turns * 1. / steps, layer=layer)
                self._nacrtaj_kocku()
                finish_time = time.perf_counter()
                standardno_vrijeme = finish_time - start_time

        return [paralelno_vrijeme, standardno_vrijeme]

    """
    Resetuj prikaz - vrati u početni položaj
    """
    def _resetuj_prikaz(self, *args):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._nacrtaj_kocku()

    def _izmijesaj_kocku(self, *args):
        print("\n\n MIJEŠAM KOCKU: \n")
        for t in range(5):
            plt.pause(0.01)
            self.cube.izmijesaj()
            self._nacrtaj_kocku()

    """
    Riješi kocku - okretanje kocke
    """
    def _rijesi_kocku(self, *args):
        print("\n\n POSTUPAK RJEŠAVANJA KOCKE : \n")
        move_list = self.cube._move_list[:]
        paralelno_vrijeme = 0
        standardno_vrijeme = 0
        for (face, n, layer) in move_list[::-1]:
            vrijeme = self.rotiraj_stranicu(face, -n, layer)   
            paralelno_vrijeme += vrijeme[0]
            standardno_vrijeme += vrijeme[1]
        
        print("\n \nPotrebno vrijeme za izvršavanje programa upotrebom paralelizma je: " + str(paralelno_vrijeme))
        print("Standardno potrebno vrijeme je: " + str(standardno_vrijeme) + "\n\n")
        self.cube._move_list = []


    """
    Handler za klik
    """
    def _key_press(self, event):
        if event.key.isdigit():
            self._digit_flags[int(event.key)] = 1
        elif event.key.upper() in 'LRUDBF':
            print("\n\n ROTIRAM KOCKU KLIKOM NA SLOVO: " + event.key.upper() + "\n")
            direction = 1
            if np.any(self._digit_flags[:N]):
                for d in np.arange(N)[self._digit_flags[:N]]:
                    self.rotiraj_stranicu(event.key.upper(), direction, layer=d)
            else:
                self.rotiraj_stranicu(event.key.upper(), direction)
                
        self._nacrtaj_kocku()

    """
    Handler za klik na slovo/cifru
    """
    def _key_release(self, event):
        if event.key.isdigit():
            self._digit_flags[int(event.key)] = 0

    """
    Handler na klik miša
    """
    def _mouse_press(self, event):
        """Handler for mouse button press"""
        self._event_xy = (event.x, event.y)
        if event.button == 1:
            self._button1 = True

    """
    Handler za puštanje miša nakon pomjeranja
    """
    def _mouse_release(self, event):
        self._event_xy = None
        if event.button == 1:
            self._button1 = False

    """Handler za pomjeranje mišem"""
    def _mouse_motion(self, event):
        if self._button1:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)

            ax_LR = self._ax_LR
            rot1 = Kvaternion.kreiraj_kvaternion_od_v_i_theta(self._ax_UD, self._step_UD * dy)
            rot2 = Kvaternion.kreiraj_kvaternion_od_v_i_theta(ax_LR, self._step_LR * dx)
            self.rotiraj(rot1 * rot2)

            self._nacrtaj_kocku()

"""
Pokretanje aplikacije, defaultna vrijednost broja N je 3 (kocka 3 x 3 x 3)
"""
if __name__ == '__main__':
    import sys
    try:
        N = int(sys.argv[1])
    except:
        N = 3

    c = Kocka(N)
    c.nacrtaj_interaktivnu()
    plt.show()

   
