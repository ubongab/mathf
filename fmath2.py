import sys
import cmath
import streamlit as st
import streamlit.web.cli as stcli
from streamlit import runtime
import numpy as np
import matplotlib.pyplot as plt

st.title('FMath A2 - Solution')

# Load dataset
data = {
    "A1": [2.58, 2.92, 2.40, 2.70, 2.00, 1.96, 2.50, 3.75, 2.40, 2.30, 3.00, 2.20, 2.10, 1.80, 1.95, 1.58, 1.00, 3.80, 1.73, 1.60],
    "A2": [4.55, 3.68, 3.70, 3.50, 4.80, 4.65, 3.90, 4.42, 4.25, 3.85, 3.70, 4.10, 4.60, 3.60, 2.80, 4.24, 2.86, 3.10, 3.85, 4.28],
    "B1": [120, 110, 100, 130, 140, 150, 145, 100, 95, 90, 200, 180, 170, 120, 80, 160, 150, 140, 100, 105],
    "B2": [120, 110, 100, 130, 140, 150, 145, 100, 95, 90, 200, 180, 170, 120, 80, 160, 150, 140, 100, 105],
    "L1": [10, 8, 5, 6, 7, 11, 12, 3, 15, 5, 6, 16, 13, 8, 4, 3, 19, 13, 10, 8],
    "L2": [28, 20, 15, 12, 18, 19, 30, 16, 28, 8, 26, 35, 18, 21, 24, 12, 38, 23, 21, 18],
    "W": [17, 15, 10, 8, 12, 16, 20, 12, 21, 6, 16, 27, 14, 15, 12, 9, 25, 19, 16, 12],
}


set_no = [i for i in range(1,21)]

def mat2latex(mat):
    mat_latex = r'$\begin{bmatrix} ' + ' \\\ '.join([' & '.join(map(str, row)) for row in mat]) + r' \end{bmatrix}$'
    return mat_latex


# Define the Streamlit app
def main():
    
    # Define matrices
    A = np.array([2, 4, 6]).reshape(1,3)
    B = np.array([[1],[3]])
    C = np.array([[3, 1], [4,2]])
    D = np.array([-1,2]).reshape(1,2)
    E = np.array([[1, 0, 2], [3,1,1], [-1,2,3]])
    F = np.array([[4, 5], [1, 2]])
    G = np.array([[2], [-1], [3]])
    H = np.array([[2, 3, -1], [1,3,1], [0,2,4]])
    I = np.array([[1, -1], [2, 4]])
    K = np.array([[3, 1, 1], [2,0,1], [-1,2,4]])
    
    # Define dataset
    dataset = {'V': [F,I,C,F,I,C,F,I,F,C,F,I], 
               'X': [E, E, H, H, K, K, E, E, H, H, K, K], 
               'Y': [H, H, E,E,E,E,K,K,K,K,H,H], 
               'Z': [A,G,G,A,A,G,G,A,A,G,A,G]
               }
    
 
    # Create Streamlit app
    
    MAX_LEN = len(dataset['V']) + 1
    data = st.sidebar.selectbox('Select Dataset', range(1,MAX_LEN))
    
    V = dataset['V'][data-1]
    X = dataset['X'][data-1]
    Y = dataset['Y'][data-1]
    Z = dataset['Z'][data-1]

    with st.expander("Question 1 ", expanded =True):
        st.subheader("1. Solution is below")
        result = {}
        
        result['a) X+Y'] = X + Y
        try:
            result['b) X.Z'] = np.dot(X, Z)
        except Exception as e:
            result['b) X.Z'] = np.array(['Bad', 'matrix'])
        try:
            result['c) Z.X'] = np.dot(Z, X) 
        except Exception as e:
            result['c) Z.X'] = np.array(['Bad', 'matrix'])
        result['d) 3X'] = 3 * X 
        result['e) 3X-Y'] = 3 * X - Y 
        
        # Display results as matrices
        st.write(f"X =  {mat2latex(X)}      Y = {mat2latex(Y)}      Z = {mat2latex(Z)}")
        st.write("$___________________________________________________________________________$")


        for ind, A in result.items():
            try:
                matrix_latex = r'$\begin{bmatrix} ' + ' \\\ '.join([' & '.join(map(str, row)) for row in A]) + r' \end{bmatrix}$'
                st.write(f"{ind} = ",matrix_latex)
            except Exception as e:
                st.write(e)
    
    with st.expander("Question 2 ", expanded =True):
        st.subheader("2. Solution is below")
        for item in [V,X,Y,Z]:
            if item.shape[0] == item.shape[1]:
                st.write(f"{mat2latex(item)} determinant = {np.linalg.det(item)}")

    with st.expander("Question 3 ", expanded =False):
        st.subheader("3. Simultaneous Equation Solution")
        #Equations
        A3 = [[2,3],[18]]
        B3 = [[1,-2],[-5]]
        C3 = [[3,1],[13]]
        D3 = [[4,-1],[8]]
        E3 = [[6,-3],[6]]
        F3 = [[5,-4],[-1]]
        # 𝐴. 2𝑥 + 3𝑦 = 18    𝐵. 𝑥 − 2𝑦 = −5     𝐶. 3𝑥 + 𝑦 = 13
        # 𝐷. 4𝑥 − 𝑦 = 8      𝐸. 6𝑥 −3𝑦 = 6   𝐹. 5𝑥 − 4𝑦 = −1

        #dataset
        eqn1 = [A3,B3,B3,B3,E3,A3,A3,B3,B3,B3,B3,C3]
        eqn2 = [F3,C3,D3,E3,F3,B3,C3,C3,D3,E3,F3,D3]

        def solve_eqn(eqn1,eqn2):
            Ax = np.array([eqn1[0],eqn2[0]])
            b = np.array([eqn1[1],eqn2[1]])
            soln = np.linalg.solve(Ax, b)
            st.write(mat2latex(Ax), mat2latex(['x','y']), '=',  mat2latex(b))
            return soln
        
        xy = solve_eqn(eqn1[data-1],eqn2[data-1])
        st.write(f"x = {xy[0][0]} , y = {xy[1][0]}")
 
    with st.expander("Question 4 ", expanded =False):
        st.subheader("4. Simultaneous Equation 3-unknowns")
        #Equations
        A3 = [[2,-2,2],[2]]
        B3 = [[-1,-1,1],[3]]
        C3 = [[3,-1,2],[3]]
        D3 = [[-1,1,2],[11]]
        E3 = [[3,1,-2],[-9]]
        F3 = [[2,3,1],[8]]
        # A. 2a-2b+2c=2      B. -a-b+c=3     C. 3a-b+2c=3    D. -a+b+2c=11 
        # E. 3a+b-2c=-9      F. 2a+3b+c=8
        #dataset
        eqn1 = [B3,B3,B3,D3,B3,D3,A3,A3,B3,B3,A3,A3]
        eqn2 = [C3,C3,C3,E3,C3,E3,B3,B3,C3,C3,B3,B3]
        eqn3 = [D3,E3,F3,F3,F3,F3,E3,F3,D3,E3,C3,D3]
        

        def solve_eqn(eqn1, eqn2, eqn3):
            Ax = np.array([eqn1[0],eqn2[0],eqn3[0]])
            constants = [eqn1[1][0], eqn2[1][0], eqn3[1][0]]
            b = np.array(constants)
            soln = np.linalg.solve(Ax, b)
            # st.write(b)
            st.write(mat2latex(Ax), mat2latex(['a','b','c']), '=', b)
            # st.write(r'$\begin{bmatrix}11\\9\\2\end{bmatrix}$')
            return soln

        def solve_crammars_rule(eqn1, eqn2, eqn3):
            

            xyz = solve_eqn(eqn1[data-1],eqn2[data-1], eqn3[data-1])
            st.write(f"a = {round(xyz[0],1)} , b = {round(xyz[1],1)} , c = {round(xyz[2],1)}")

    with st.expander("Question 5,6,7,8 ", expanded =True):
        st.subheader("Question 5,6,7,8 Complex Numbers ")
        #Equations
        A, B = 2+3j, 3+2j
        C, D = 3-2j, 1-2j
        E, F = -3+2j, -2+1j
        
        # A=2+j3 B=3+j2 C=3-j2
        # D=1-j2 E=-3+j2 F=-2+j1
        #dataset
        X = [A,B,B,B,C,D,D,D,D,D,B,B]
        Y = [F,A,C,D,F,A,B,C,E,F,E,F]
        
        st.write(rf"$X = {X[data-1]}$")
        st.write(rf"$Y = {Y[data-1]}$")
        q5a = X[data-1] + Y[data-1]
        q5b = Y[data-1] - X[data-1]
        st.write(rf"5a) $X + Y = {q5a}$")
        st.write(rf"5b) $Y - X = {q5b}$")
        
        #-------------------- 6a & 6b ------------------
        Xr, Xtheta = cmath.polar(X[data-1])
        Yr, Ytheta = cmath.polar(Y[data-1])
        Xr_i, Xtheta_i = cmath.polar(q5a)
        Yr_i, Ytheta_i = cmath.polar(q5b)
        st.write(f"6a) $X_polar = {Xr:.2f}<{Xtheta*180/np.pi:.2f}$")
        st.write(f"6b) $Y_polar = {Yr:.2f}<{Ytheta*180/np.pi:.2f}$")

        st.write(f"6ai) $X_plus_Y_polar = {Xr_i:.2f}<{Xtheta_i*180/np.pi:.2f}$")
        st.write(f"6bi) $Y_minus_X_polar = {Yr_i:.2f}<{Ytheta_i*180/np.pi:.2f}$")

        # st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.plot(q5a.real, q5a.imag, 'o')
        plt.plot(q5b.real, q5b.imag, 's')
        plt.text(q5a.real+0.01,q5a.imag-0.1,f"Q5a {q5a.real,q5a.imag}", color="g")
        plt.text(q5b.real-0.6,q5b.imag+0.1,f"$Q5b {q5b}$", color='blue')
        st.pyplot()
        #-------------------- 7a ------------------
        r = Xr*Yr
        deg = Xtheta*180/np.pi + Ytheta*180/np.pi
        st.write(f"7a) $X.Y = {r:.2f}<{deg:.2f}$")
        
        # ------------------ 7b --------------------
        r = Xr/Yr
        deg = Xtheta*180/np.pi - Ytheta*180/np.pi
        st.write(f"7b) $X/Y = {r:.2f}<{deg:.2f}$")

        # ------------------ 8a & 8b --------------------
        st.write(r"Polar2Cartesian:=  $r\cos(\theta) + r\sin(\theta)$")
        st.write(f"8a) cartesian: $X.Y = {X[data-1]*Y[data-1]:.2f}$")
        st.write(f"8b) cartesian: $X/Y = {X[data-1]/Y[data-1]:.2f}$")
        
    with st.expander("Question 9,10,11 ", expanded =True):
        st.subheader("Question 9,10,11 Complex Numbers ")
        #Equations
        A, B = 3+4j, 4-5j
        C, D = 5-2j, 1-3j
        E, F = 2+3j, 4+1j
        
        # A =3+j4 B=4-j5 C=5-j2 D=1-j3
        # E=2+j3 F=4+j1
        #dataset
        V = [A,B,B,B,C,D,D,D,D,D,B,B]
        W = [F,A,C,D,F,A,B,C,E,F,E,F]
        
        st.write(rf"$V = {V[data-1]}$")
        st.write(rf"$W = {W[data-1]}$")
        q9 = V[data-1] + W[data-1]
        q10 = V[data-1] - W[data-1]
        st.write(rf"9) $V + W = {q9}$")
        st.write(rf"10) $V - W = {q10}$")
        # --------------- Argand diagram -------------------------
        plt.plot(q9.real, q9.imag, 'o')
        plt.text(q9.real-1.3,q9.imag-0.1,f"$Q9 {q9}$", color="g")
        plt.plot(q10.real, q10.imag, 's')
        plt.text(q10.real+0.01,q10.imag+0.1,fr"$Q10 {q10}$", color='blue')
        st.pyplot()

        #-------------------- 11 ----------------------------------
        # V = 12<0
        Z1r, Z1theta = cmath.polar(V[data-1])
        Z2r, Z2theta = cmath.polar(W[data-1])
        Zr, Ztheta = cmath.polar(q9)
        
        
        st.write(rf"11) $V_rect = {Z1r:.2f}<{Z1theta*180/np.pi:.2f}$")
        st.write(rf"11) $W_rect = {Z2r:.2f}<{Z2theta*180/np.pi:.2f}$")
        st.write(rf"11) $Z_Trect = {q9}$")
        st.write(f"11) $ Z_Tpolar = {Zr:.2f}<{Ztheta*180/np.pi:.2f}$")
        st.write(f"11) $V_polar = 12<0$")
        # st.write(rf"11) $Z_T = {12}<0$")
        # ------------------ 11 polar form --------------------
        Vr, Vtheta = 12, 0
        r = Vr/Zr
        deg = Vtheta - Ztheta*180/np.pi
        st.write(f"11) $V/Z = {r:.2f}<{deg:.2f}$")
         


# Run the Streamlit app
if __name__ == '__main__':
    # if runtime.exists():
    main()
    # else:
    #     sys.argv = ["streamlit", "run", sys.argv[0]]
    #     sys.exit(stcli.main())
