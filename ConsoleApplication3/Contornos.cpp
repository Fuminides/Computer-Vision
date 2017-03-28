#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define OTSU false


//Structs
struct Gradiente {
	double x, y, norma, theta;
} gradiente;

int filtro_sobel_y[3][3] = {
	-1,0,1,
	-2,0,2,
	-1,0,1
};

int filtro_sobel_x[3][3] = {
	-1,-2,-1,
	0,0,0,
	1,2,1
};
//Variables

//Funciones
double aplicar_filtro_sobel(Mat imagen, int x, int y, int filtro[3][3]) {
	double negativos, acum_positivos;
	if (filtro[0][1] == 0) { //Filtro para y
		negativos = filtro[0][0] * imagen.at<unsigned char>(x - 1, y - 1) +
			filtro[1][0] * imagen.at<unsigned char>(x - 1, y) +
			filtro[2][0] * imagen.at<unsigned char>(x - 1, y + 1);
		acum_positivos = filtro[0][2] * imagen.at<unsigned char>(x + 1, y - 1) +
			filtro[1][2] * imagen.at<unsigned char>(x + 1, y) +
			filtro[2][2] * imagen.at<unsigned char>(x + 1, y + 1);

		return (negativos + acum_positivos) / 4.0;
	}
	else { //Filtro para y
		acum_positivos = filtro[0][0] * imagen.at<unsigned char>(x - 1, y - 1) +
			filtro[0][1] * imagen.at<unsigned char>(x , y - 1) +
			filtro[0][2] * imagen.at<unsigned char>(x + 1, y - 1);
		negativos = filtro[2][0] * imagen.at<unsigned char>(x - 1, y + 1) +
			filtro[2][1] * imagen.at<unsigned char>(x, y + 1) +
			filtro[2][2] * imagen.at<unsigned char>(x + 1, y + 1);
		return (negativos + acum_positivos) / 4.0;
	}
}

Gradiente operatorSobel(Mat imagen, int x, int y) {
	double Gx = aplicar_filtro_sobel(imagen, x,y,filtro_sobel_x);
	double Gy = aplicar_filtro_sobel(imagen, x, y, filtro_sobel_y);
	Gradiente gradiente_calculado;
	gradiente_calculado.x = Gx;
	gradiente_calculado.y = Gy;
	gradiente_calculado.norma = sqrt(Gx*Gx+Gy*Gy);
	gradiente_calculado.theta = atan2(Gy,Gx);

	return gradiente_calculado;
}


//MAIN
void main(int argc, char ** argv) {
	Mat imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);	
	Mat GX(imagen.size[0], imagen.size[1], CV_8U), GY(imagen.size[0], imagen.size[1], CV_8U),
		Theta(imagen.size[0], imagen.size[1], DataType<double>::type),
		Theta_dibujar(imagen.size[0], imagen.size[1], DataType<double>::type),
		Modulos(imagen.size[0], imagen.size[1], CV_32F);
	vector<int> puntos_votacion(imagen.size[1],0);
	Mat grad_x, grad_y, Modulos_dibujar(imagen.size[0], imagen.size[1], CV_8U);
	Mat abs_grad_x, abs_grad_y;

	imshow("Original", imagen);
	if (strcmp(argv[2], "-sa") == 0) {
		GaussianBlur(imagen, imagen, Size(3, 3), 0, 0, BORDER_DEFAULT);
		int imgX = imagen.size[0], imgY = imagen.size[1];
		for (int i = 1; i < imgX - 1; i++) {
			for (int z = 1; z < imgY; z++) {
				Gradiente aux = operatorSobel(imagen, i, z);

				GX.at<unsigned char>(i, z) = (aux.x / 2.0 + 128);
				GY.at<unsigned char>(i, z) = (aux.y / 2.0 + 128);
				Theta.at<double>(i, z) = aux.theta;
				Modulos.at<double>(i, z) = aux.norma;
				Modulos_dibujar.at<unsigned char>(i, z) = aux.norma;
			}
		}
		
	}
	else if (strcmp(argv[2], "-s") == 0) {
		GaussianBlur(imagen, imagen, Size(3, 3), 0, 0, BORDER_DEFAULT);
		Sobel(imagen, grad_x, CV_32F, 1, 0, 3);
		Sobel(imagen, grad_y, CV_32F, 0, 1, 3);
		grad_x = grad_x / 4.0;
		grad_y = grad_y / 4.0;

		Mat GX_aux = grad_x / 2.0 + 128;
		Mat GY_aux = grad_y /2.0 + 128;

		phase(grad_x, grad_y, Theta);
		magnitude(grad_x, grad_y, Modulos);
		Modulos_dibujar = Modulos.clone();
		Modulos_dibujar.convertTo(Modulos_dibujar, CV_8U);
		GX_aux.convertTo(GX, CV_8U);
		GY_aux.convertTo(GY, CV_8U);
	}
	
	imshow("Contornos GX", GX);
	imshow("Contornos GY", GY);
	imshow("Contornos Theta", Theta );
	imshow("Contornos Modulos", Modulos_dibujar);
	int numFilas = Modulos.size[0], numColumnas = Modulos.size[1];
	cout << numFilas << endl;
	for (int i = 0; i < numFilas; ++i) {
		for (int j = 0; j < numColumnas; ++j) {
			if (Modulos_dibujar.at<unsigned char>(i, j) > 70) {
				int x = j - numColumnas / 2;
				int y = numFilas / 2 - i;
				double p = x*cos(Theta.at<float>(i, j)) + y*sin(Theta.at<float>(i, j));
				if ((abs(cos(Theta.at<float>(i, j))) <= 0.95) && (abs(cos(Theta.at<float>(i, j))) >= 0.05)) {
					int eleccion = p / cos(Theta.at<float>(i, j));
					//cout << "Votamos: " << to_string(eleccion + numColumnas / 2) << endl;
					if (((eleccion + numColumnas / 2) >= 0) && ((eleccion + numColumnas / 2) < numColumnas)) puntos_votacion[eleccion + numColumnas / 2]++;
				}
			}
		}
	}
	int max_value = 0, best_index;
	for (int i = 0; i < imagen.size[1]; i++) {
		if (puntos_votacion[i] > max_value) {
			max_value = puntos_votacion[i];
			best_index = i;
		}
	}
	cvtColor(imagen, imagen, CV_GRAY2BGR);
	Point mejor(best_index, numFilas / 2);
	circle(imagen, mejor, 5, Scalar(255, 0, 0));
	imshow("Punto de fuga", imagen);


	cout << numFilas / 2 << " " << best_index << endl;
	
	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape

}