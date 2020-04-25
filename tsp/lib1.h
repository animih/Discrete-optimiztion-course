class Matrix;

class Vector{
	double x;
	double y;

	friend Vector operator+(const Vector &, const Vector &);
	friend Vector operator-(const Vector &, const Vector &);
	friend std::ostream &operator<<(std::ostream & out, const Vector & A);
	friend std::istream &operator>>(std::istream & in, Vector & A);

	public:
		Vector(double , double);
		Vector();

};

class Matrix{

}

Vector::Vector(double x, double y){
	this->x = x;
	this->y = y;
}
Vector::Vector(){
	this->x = 0;
	this->y = 0;
}

Vector operator+(const Vector &A, const Vector &B){
	return Vector(A.x+B.x, A.y+B.y);
}

Vector operator-(const Vector &A, const Vector &B){
	return Vector(A.x-B.x, A.y-B.y);
}