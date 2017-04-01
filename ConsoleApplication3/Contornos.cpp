#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define TAM_CANNY 5
#define E 2.71828182
#define SIGMA 1

//Structs
struct Gradiente {
	double x, y, norma, theta;
} gradiente;

//Variables
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

double filtro_canny_x[TAM_CANNY];
double filtro_canny_x_gaussian[TAM_CANNY];
double filtro_canny_y[TAM_CANNY];
double filtro_canny_y_gaussian[TAM_CANNY];

double normalizacion_canny = 0, normalizacion_canny_gaussian=0;

double termino_gaussiana(double sigma, double x, double i) {
	return exp(-(x*x)/(2*sigma*sigma)) ;
}

double termino_canny(double sigma, double x) {
	return -x / (sigma*sigma) * exp(-x*x/(2*sigma*sigma));
}

void generar_cannny(double sigma) {
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2;
		double calculo_termino = termino_canny(SIGMA, indice);
		if (calculo_termino > 0) normalizacion_canny += calculo_termino;
		filtro_canny_x[i] = -calculo_termino;
		filtro_canny_y[i] = calculo_termino;
		
		calculo_termino = termino_gaussiana(SIGMA,indice,1);
		if (calculo_termino > 0) normalizacion_canny_gaussian += calculo_termino;
		filtro_canny_x_gaussian[i] = calculo_termino;
		filtro_canny_y_gaussian[i] = calculo_termino;
	}
	/*for (int i = 0; i < TAM_CANNY; i++) {
		cout << filtro_canny_y[i] << " ";
	}
	cout << endl;*/
}

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

void operatorCannyX(Mat imagen, int x, int y, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + x;
		//cout << "Indice: " << indice << " y: " << y << endl;
		result += imagen.at<unsigned char>(indice, y)*filtro[i];
	}
	resultado.at<double>(x, y) = result;
}

void operatorCannyY(Mat imagen, int fila, int columna, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + columna;
		result += imagen.at<unsigned char>(fila, indice)*filtro[i];
	}
	resultado.at<double>(fila, columna) = result;
}

void operatorCannyXf(Mat imagen, int x, int y, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + x;
		//cout << "Indice: " << indice << " y: " << y << endl;
		result += imagen.at<double>(indice, y)*filtro[i];
	}
	resultado.at<double>(x, y) = result;
}

void operatorCannyYf(Mat imagen, int fila, int columna, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + columna;
		result += imagen.at<double>(fila, indice)*filtro[i];
	}
	resultado.at<double>(fila, columna) = result;
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

void dibujarFuga(Mat imagen) {
	Mat GX(imagen.size[0], imagen.size[1], CV_8U), GY(imagen.size[0], imagen.size[1], CV_8U),
		Theta(imagen.size[0], imagen.size[1], DataType<double>::type),
		Theta_dibujar(imagen.size[0], imagen.size[1], DataType<double>::type),
		Modulos(imagen.size[0], imagen.size[1], CV_32F);
	vector<vector<int>> puntos_votacion(imagen.size[0], std::vector<int>(imagen.size[1], 0));
	Mat grad_x, grad_y, Modulos_dibujar(imagen.size[0], imagen.size[1], CV_8U);
	Mat abs_grad_x, abs_grad_y;

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
	imshow("Contornos GX", GX);
	imshow("Contornos GY", GY);
	imshow("Contornos Theta", Theta);
	imshow("Contornos Modulos", Modulos_dibujar);
	int numFilas = Modulos.size[0], numColumnas = Modulos.size[1];

	for (int i = 0; i < numFilas; ++i) {
		for (int j = 0; j < numColumnas; ++j) {
			if (Modulos_dibujar.at<unsigned char>(i, j) > 70) {
				int x = j - numColumnas / 2;
				int y = numFilas / 2 - i;
				double p = x*cos(Theta.at<float>(i, j)) + y*sin(Theta.at<float>(i, j));

				if ((abs(cos(Theta.at<float>(i, j))) <= 0.95) && (abs(cos(Theta.at<float>(i, j))) >= 0.05)) {

					for (int elecciones = 0; elecciones < numFilas; ++elecciones) {
						int valor_y = elecciones - numFilas / 2;
						int eleccion = p - valor_y*sin(Theta.at<float>(i, j)) / cos(Theta.at<float>(i, j)); //A partir de la y, despejo de la x
						if (((eleccion + numColumnas / 2) >= 0) && ((eleccion + numColumnas / 2) < numColumnas)) {
							//cout << "Votamos: [" << to_string(eleccion + numColumnas / 2) << ", " <<elecciones + numFilas / 2 <<"] " << puntos_votacion[eleccion + numColumnas / 2][elecciones + numFilas / 2] <<endl;
							puntos_votacion[eleccion + numColumnas / 2][elecciones]+=1;
						}
					}
				}
			}
		}
	}
	int max_value = 0, best_index_x, best_index_y;
	for (int i = 0; i < imagen.size[0]; i++) {
		for (int z = 0; z < imagen.size[1]; z++) {
			if (puntos_votacion[i][z] > max_value) {
				max_value = puntos_votacion[i][z];
				cout << max_value << "i: "<< i << ", z:" << z << endl;
				best_index_x = i;
				best_index_y = z;
			}
		}
	}
	cvtColor(imagen, imagen, CV_GRAY2BGR);
	Point mejor(best_index_x, best_index_y);
	drawMarker(imagen, mejor, cv::Scalar(0, 0, 255), MARKER_CROSS, 15, 1);
	imshow("Punto de fuga", imagen);


	cout << " [" << best_index_x << ", " << best_index_y << "]" << endl;

	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape
}


//MAIN
void main(int argc, char ** argv) {
	Mat imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);	
	//dibujarFuga(imagen);
	//exit(0);
	Mat GX(imagen.size[0], imagen.size[1], CV_8U), GY(imagen.size[0], imagen.size[1], CV_8U),
		Theta(imagen.size[0], imagen.size[1], DataType<double>::type),
		Theta_dibujar(imagen.size[0], imagen.size[1], DataType<double>::type),
		Modulos(imagen.size[0], imagen.size[1], CV_64F);
	vector<int> puntos_votacion(imagen.size[1],0);
	Mat grad_x(imagen.size[0], imagen.size[1], CV_64F), 
		grad_y(imagen.size[0], imagen.size[1], CV_64F), 
		Modulos_dibujar(imagen.size[0], imagen.size[1], CV_8U);
	Mat abs_grad_x(imagen.size[0], imagen.size[1], CV_64F), 
		abs_grad_y(imagen.size[0], imagen.size[1], CV_64F);

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
				Modulos_dibujar.convertTo(Modulos_dibujar, CV_8U);
				GX.convertTo(GX, CV_8U);
				GY.convertTo(GY, CV_8U);

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
	else if (strcmp(argv[2], "-c") == 0) {
		generar_cannny(SIGMA);
		int filas = imagen.size[0], columnas = imagen.size[1];
		/*for (int i = 0; i < TAM_CANNY; i++) {
			cout << filtro_canny_y[i] << " ";
		}
		cout << endl;*/

		for (int i = TAM_CANNY; i < filas - TAM_CANNY; i++) {
			for (int z = TAM_CANNY; z < columnas - TAM_CANNY; z++) {
				operatorCannyY(imagen, i, z, filtro_canny_x_gaussian, grad_x);
				operatorCannyY(imagen, i, z, filtro_canny_y, grad_y);
			}
		}

		grad_x = grad_x / normalizacion_canny_gaussian;
		grad_y = grad_y / normalizacion_canny;

		for (int i = TAM_CANNY; i < filas - TAM_CANNY; i++) {
			for (int z = TAM_CANNY; z < columnas - TAM_CANNY; z++) {
				operatorCannyXf(grad_x, i, z, filtro_canny_x, abs_grad_x);
				operatorCannyXf(grad_y, i, z, filtro_canny_y_gaussian, abs_grad_y);
			}
		}
		abs_grad_x = abs_grad_x / normalizacion_canny;
		abs_grad_y = abs_grad_y / normalizacion_canny_gaussian;
		cout << abs_grad_x.size() << endl;

		//cout << imagen << endl;
		Mat GX_aux = abs_grad_x / 2.0 + 128;
		Mat GY_aux = abs_grad_y / 2.0 + 128;
		cout << GX_aux.size() << endl;

		phase(abs_grad_x, abs_grad_y, Theta);
		magnitude(abs_grad_x, abs_grad_y, Modulos);
		Modulos_dibujar = Modulos.clone();
		Modulos_dibujar.convertTo(Modulos_dibujar, CV_8U);
		GX_aux.convertTo(GX, CV_8U);
		GY_aux.convertTo(GY, CV_8U);
		cout << GX.size() << endl;

	}
	
	imshow("Contornos GX", GX);
	imshow("Contornos GY", GY);
	imshow("Contornos Theta", Theta );
	imshow("Contornos Modulos", Modulos_dibujar);
	int numFilas = Modulos.size[0], numColumnas = Modulos.size[1];

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
	cout << "Punto de fuga: " << mejor << endl;
	drawMarker(imagen, mejor, cv::Scalar(0, 0, 255), MARKER_CROSS, 15, 1);
	imshow("Punto de fuga", imagen);


	cout << numFilas / 2 << " " << best_index << endl;
	
	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape

}

