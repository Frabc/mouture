# Author: David Roubinet
# Date:   Aug25
import math

class Vec():
    """3D vectors defined by x,y,z coordinates"""
    def __init__(self,x:float,y:float,z:float):
        (self.x,self.y,self.z) = (float(x),float(y),float(z))
    def __iter__(self):  A=self; return iter([A.x,A.y,A.z])
    def __sub__(self,B): A=self; return Vec(A.x-B.x,A.y-B.y,A.z-B.z)
    def __add__(self,B): A=self; return Vec(A.x+B.x,A.y+B.y,A.z+B.z)
    def norm(self):      A=self; return (A.x*A.x+A.y*A.y+A.z*A.z)**0.5
    # homothecy product : vector * scalar -> vector
    def scale(self,k):   A=self; return Vec(k*A.x,k*A.y,k*A.z)
    # dot product : vector * vector -> scalar
    def dot(self,B):     A=self; return A.x*B.x+A.y*B.y+A.z*B.z
    # cross product : vector * vector -> vector
    def cross(self,B):   A=self; return Vec( A.y*B.z-A.z*B.y ,
                                           -(A.x*B.z-A.z*B.x),
                                             A.x*B.y-A.y*B.x )

class Mtx():
    """3D transformation matrix defined by columns"""
    def  __init__(self,col0:Vec,col1:Vec,col2:Vec):
        self.col0,self.col1,self.col2 = (col0,col1,col2)
    def inv(self):
        # Assign names to coefficients
        (m00,m10,m20) = self.col0
        (m01,m11,m21) = self.col1
        (m02,m12,m22) = self.col2
        # Compute all cofactors
        cf00=m11*m22-m12*m21; cf10=m01*m22-m02*m21; cf20=m01*m12-m02*m11
        cf01=m10*m22-m12*m20; cf11=m00*m22-m02*m20; cf21=m00*m12-m02*m10
        cf02=m10*m21-m11*m20; cf12=m00*m21-m01*m20; cf22=m00*m11-m01*m10
        # Compute Determinant along first row
        det= m00*cf00 - m01*cf01 + m02*cf02
        # Return transposed comatrix divided by determinant
        return Mtx( Vec( cf00/det,-cf01/det, cf02/det),
                    Vec(-cf10/det, cf11/det,-cf12/det),
                    Vec( cf20/det,-cf21/det, cf22/det) )

    def apply(self,A:Vec):
        # Assign names to coefficients
        (m00,m10,m20) = self.col0
        (m01,m11,m21) = self.col1
        (m02,m12,m22) = self.col2
        (  x,  y,  z) = A
        # Multiply matrix * vector -> vector
        return Vec(m00*x+m01*y+m02*z,
                   m10*x+m11*y+m12*z,
                   m20*x+m21*y+m22*z)

if __name__=="__main__":
    # Unit testing vectors
    A = Vec(2,3,-5.4)
    assert (A.x,A.y,A.z)==(2,3,-5.4)
    size=A.norm()
    A=A.scale(2)
    assert (A.x,A.y,A.z)==(4,6,-10.8)
    assert A.norm()==2*size
    A = Vec(1,2,3)
    B = Vec(4,0,6)
    assert A.dot(B)==1*4+2*0+3*6
    C=A.cross(B)
    assert list(C)==[12,6,-8]
    assert A.dot(C)==0
    assert B.dot(C)==0
    # Unit testing transforms
    θ = 2*math.pi/6
    # transformation scale x and rotate around x
    T = Mtx( Vec(2,          0,           0) ,
             Vec(0, math.cos(θ),math.sin(θ)) ,
             Vec(0,-math.sin(θ),math.cos(θ)) )
    A1 = T.apply(A)
    A2 = T.inv().apply(A1)
    assert (A2-A).norm() < 0.001
    print("Selftest PASS")
