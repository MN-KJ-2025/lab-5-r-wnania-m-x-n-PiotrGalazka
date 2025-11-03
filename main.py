# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
from typing import Union

def spare_matrix_Abt(m: int, n: int) -> Union[tuple[np.ndarray, np.ndarray], None]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,n) i
    wektora b (m,) na podstawie pomocniczego wektora t (m,).

    Args:
        m (int): Liczba wierszy macierzy A.
        n (int): Liczba kolumn macierzy A.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A o rozmiarze (m,n),
            - Wektor b (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or not isinstance(m, int) or n<=0 or m<=0:
        return None

    t = np.linspace(0,1,m)
    A = np.vander(t,n,True)
    b = np.array([np.cos(4*i) for i in t ])
    return A , b



def square_from_rectan(
    A: np.ndarray, b: np.ndarray
) -> Union[tuple[np.ndarray, np.ndarray], None]:
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników
    na kwadratowy układ równań.
    A^T * A * x = A^T * b  ->  A_new * x = b_new

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A_new o rozmiarze (n,n),
            - Wektor b_new (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A , np.ndarray) or not isinstance(b, np.ndarray) or len(A)!= len(b) or np.ndim(b) != 1 or np.ndim(A)!=2:
        return None 
    
    A_new = A.transpose()@A
    b_new = A.transpose()@b

    return A_new,b_new


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> Union[float, None]:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray)) or not isinstance(x, (np.ndarray)) or not isinstance(b, (np.ndarray))or np.ndim(b) != 1 or np.ndim(A)!=2 or np.ndim(x) != 1 or len(A)!=len(b):
        return None
    
    (i,j) = np.shape(A)
    if len(x)!= j:
        return None  
      
    return np.linalg.norm(b-A@x)
