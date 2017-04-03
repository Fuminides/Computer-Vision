#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define TAM_CANNY 5
#define E 2.71828182
#define PI 3.1416
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

//----Funciones Canny----
double filtro_canny_x[TAM_CANNY];
double filtro_canny_y[TAM_CANNY];
double filtro_canny_gaussian[TAM_CANNY];

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
		filtro_canny_gaussian[i] = calculo_termino;
		
	}
	
}

void operatorCannyY(Mat imagen, int x, int y, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + x;
		//cout << "Indice: " << indice << " y: " << y << endl;
		result += imagen.at<unsigned char>(indice, y)*filtro[i];
	}
	resultado.at<double>(x, y) = result;
}

void operatorCannyX(Mat imagen, int fila, int columna, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + columna;
		result += imagen.at<unsigned char>(fila, indice)*filtro[i];
	}
	resultado.at<double>(fila, columna) = result;
}

void operatorCannyYf(Mat imagen, int x, int y, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + x;
		//cout << "Indice: " << indice << " y: " << y << endl;
		result += imagen.at<double>(indice, y)*filtro[i];
	}
	resultado.at<double>(x, y) = result;
}

void operatorCannyXf(Mat imagen, int fila, int columna, double filtro[TAM_CANNY], Mat resultado) {
	double result = 0;
	for (int i = 0; i < TAM_CANNY; i++) {
		int indice = i - TAM_CANNY / 2 + columna;
		result += imagen.at<double>(fila, indice)*filtro[i];
	}
	resultado.at<double>(fila, columna) = result;
}

void aplicarCanny(Mat imagen, Mat grad_x, Mat grad_y, Mat abs_grad_x, Mat abs_grad_y, Mat Theta, Mat Modulos, Mat Modulos_dibujar, Mat GX, Mat GY) {
	generar_cannny(SIGMA);
	int filas = imagen.size[0], columnas = imagen.size[1];
	
	for (int i = TAM_CANNY; i < filas - TAM_CANNY; i++) {
		for (int z = TAM_CANNY; z < columnas - TAM_CANNY; z++) {
			operatorCannyY(imagen, i, z, filtro_canny_gaussian, grad_x);
			operatorCannyY(imagen, i, z, filtro_canny_y, grad_y);
		}
	}

	grad_x = grad_x / normalizacion_canny_gaussian;
	grad_y = grad_y / normalizacion_canny;

	for (int i = TAM_CANNY; i < filas - TAM_CANNY; i++) {
		for (int z = TAM_CANNY; z < columnas - TAM_CANNY; z++) {
			operatorCannyXf(grad_x, i, z, filtro_canny_x, abs_grad_x);
			operatorCannyXf(grad_y, i, z, filtro_canny_gaussian, abs_grad_y);
			double angX = abs_grad_x.at<double>(i, z);
			double angY = abs_grad_y.at<double>(i, z);
			Theta.at<double>(i, z) = atan2(angY, angX);
			Modulos.at<double>(i, z) = sqrt(angX*angX + angY*angY);
			Modulos_dibujar.at<unsigned char>(i, z) = sqrt(angX*angX + angY*angY);

		}
	}
	abs_grad_x = abs_grad_x / normalizacion_canny;
	abs_grad_y = abs_grad_y / normalizacion_canny_gaussian;

	Mat GX_aux = abs_grad_x / 2.0 + 128;
	Mat GY_aux = abs_grad_y / 2.0 + 128;

	GX_aux.convertTo(GX, CV_8U);
	GY_aux.convertTo(GY, CV_8U);
}

//---Funciones Sobel-----
double aplicar_filtro_sobel(Mat imagen, int x, int y, int filtro[3][3]) {
	double negativos, acum_positivos;
	if (filtro[0][1] == 0) { //Filtro para x
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

void aplicarSobelManual(Mat imagen, Mat grad_x, Mat grad_y, Mat Theta, Mat Modulos, Mat Modulos_dibujar, Mat GX, Mat GY) {
	GaussianBlur(imagen, imagen, Size(5, 5), 0, 0, BORDER_DEFAULT);
	int imgX = imagen.size[0], imgY = imagen.size[1];
	for (int i = 1; i < imgX - 1; i++) {
		for (int z = 1; z < imgY; z++) {
			Gradiente aux = operatorSobel(imagen, i, z);
			grad_x.at<double>(i, z) = aux.x;
			grad_y.at<double>(i, z) = aux.y;
			GX.at<unsigned char>(i, z) = (aux.x / 2.0 + 128);
			GY.at<unsigned char>(i, z) = (aux.y / 2.0 + 128);
			Theta.at<double>(i, z) = aux.theta;
			Modulos.at<double>(i, z) = aux.norma;
			Modulos_dibujar.at<unsigned char>(i, z) = aux.norma;
			

		}
	}
	GX.convertTo(GX, CV_8U);
	GY.convertTo(GY, CV_8U);
}

void aplicarSobelOpenCV(Mat imagen, Mat grad_x, Mat grad_y, Mat Theta, Mat Modulos, Mat Modulos_dibujar, Mat GX, Mat GY) {
	GaussianBlur(imagen, imagen, Size(5, 5), 0, 0, BORDER_DEFAULT);
	Sobel(imagen, grad_x, CV_64F, 1, 0, 3);
	Sobel(imagen, grad_y, CV_64F, 0, 1, 3);
	grad_x = grad_x / 4.0;
	grad_y = grad_y / 4.0;

	Mat GX_aux = grad_x / 2.0 + 128;
	Mat GY_aux = grad_y / 2.0 + 128;

	int imgX = imagen.size[0], imgY = imagen.size[1];
	for (int i = 1; i < imgX - 1; i++) {
		for (int z = 1; z < imgY - 1; z++) {
			double angX = grad_x.at<double>(i, z);
			double angY = grad_y.at<double>(i, z);
			Theta.at<double>(i, z) = atan2(angY, angX);
			Modulos.at<double>(i, z) = sqrt(angX*angX + angY*angY);
			Modulos_dibujar.at<unsigned char>(i, z) = sqrt(angX*angX + angY*angY);
		}
	}
	GX_aux.convertTo(GX, CV_8U);
	GY_aux.convertTo(GY, CV_8U);
}

//Apartado 3
void dibujarFuga(Mat imagen) {
	Mat GX(imagen.size[0], imagen.size[1], CV_8U), GY(imagen.size[0], imagen.size[1], CV_8U),
		Theta(imagen.size[0], imagen.size[1], DataType<double>::type),
		Theta_dibujar(imagen.size[0], imagen.size[1], DataType<double>::type),
		Modulos(imagen.size[0], imagen.size[1], DataType<float>::type),
		Votos(imagen.size[0], imagen.size[1], CV_8U);
	vector<vector<int>> puntos_votacion(imagen.size[0], std::vector<int>(imagen.size[1], 0));
	Mat grad_x, grad_y, Modulos_dibujar(imagen.size[0], imagen.size[1], CV_8U);
	Mat abs_grad_x, abs_grad_y;

	GaussianBlur(imagen, imagen, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Sobel(imagen, grad_x, CV_32F, 1, 0, 3);
	Sobel(imagen, grad_y, CV_32F, 0, 1, 3);
	grad_x = grad_x / 4.0;
	grad_y = grad_y / 4.0;

	Mat GX_aux = grad_x / 2.0 + 128;
	Mat GY_aux = grad_y / 2.0 + 128;

	phase(grad_x, grad_y, Theta);
	magnitude(grad_x, grad_y, Modulos);
	Modulos.convertTo(Modulos_dibujar, CV_8U);
	GX_aux.convertTo(GX, CV_8U);
	GY_aux.convertTo(GY, CV_8U);
	cv::imshow("Contornos GX", GX);
	cv::imshow("Contornos GY", GY);
	cv::imshow("Contornos Theta", Theta / PI * 128);
	cv::imshow("Contornos Modulos", Modulos_dibujar);
	int numFilas = Modulos.size[0], numColumnas = Modulos.size[1];

	for (int i = 0; i < numFilas; ++i) {
		for (int j = 0; j < numColumnas; ++j) {
			Votos.at<unsigned char>(i,j) = 0;
			if (Modulos.at<float>(i, j) > 40) {
				int x = j - numColumnas / 2;
				int y = numFilas / 2 - i;
				double angulo_positivo = Theta.at<double>(i, j);
				if (angulo_positivo < 0) angulo_positivo += 2 * PI;
				double p = x*cos(angulo_positivo) + y*sin(angulo_positivo);

				if (abs(cos(angulo_positivo) > 0.05) && (abs(cos(angulo_positivo) < 0.95))) {
					//cout << abs(cos(Theta.at<double>(i, j))) << endl;
					Votos.at<unsigned char>(i, j) = 255;
					for (int elecciones = 0; elecciones < numFilas; ++elecciones) {
						int valor_y = elecciones - numFilas / 2;
						int eleccion = p - valor_y*sin(Theta.at<double>(i, j)) / cos(Theta.at<double>(i, j)); //A partir de la y, despejo de la x
						if (((eleccion + numColumnas / 2) >= 0) && ((eleccion + numColumnas / 2) < numColumnas)) {
							//cout << "Votamos: [" << to_string(eleccion + numColumnas / 2) << ", " <<elecciones + numFilas / 2 <<"] " << puntos_votacion[eleccion + numColumnas / 2][elecciones + numFilas / 2] <<endl;
							puntos_votacion[eleccion + numColumnas / 2][elecciones]+=1;
						}
					}
				}
			}
		}
	}
	cv::imshow("Votos", Votos);
	int max_value = 0, best_index_x, best_index_y;
	for (int i = 0; i < imagen.size[0]; i++) {
		for (int z = 0; z < imagen.size[1]; z++) {
			if (puntos_votacion[i][z] > max_value) {
				max_value = puntos_votacion[i][z];
				std::cout << max_value << "i: "<< i << ", z:" << z << endl;
				best_index_x = i;
				best_index_y = z;
			}
		}
	}
	cvtColor(imagen, imagen, CV_GRAY2BGR);
	Point mejor(best_index_x, best_index_y);
	cv::drawMarker(imagen, mejor, cv::Scalar(0, 0, 255), MARKER_CROSS, 15, 1);
	cv::imshow("Punto de fuga", imagen);


	std::cout << " [" << best_index_x << ", " << best_index_y << "]" << endl;

	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape
}


//MAIN
void main(int argc, char ** argv) {
	Mat imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	Mat GX(imagen.size[0], imagen.size[1], CV_8U), GY(imagen.size[0], imagen.size[1], CV_8U),
		Votos(imagen.size[0], imagen.size[1], CV_8U),
		Modulos_dibujar(imagen.size[0], imagen.size[1], CV_8U);

	Mat grad_x(imagen.size[0], imagen.size[1], CV_64F), 
		grad_y(imagen.size[0], imagen.size[1], CV_64F),
		abs_grad_x(imagen.size[0], imagen.size[1], CV_64F),
		abs_grad_y(imagen.size[0], imagen.size[1], CV_64F),
		Theta(imagen.size[0], imagen.size[1], DataType<double>::type),
		Modulos(imagen.size[0], imagen.size[1], DataType<double>::type);
		
	vector<int> puntos_votacion(imagen.size[1], 0);

	int simbolo_eje_y = 1;
	double umbral_modulo, cos_min, cos_max;

	cv::imshow("Original", imagen);

	if (strcmp(argv[2], "-sa") == 0) {
		simbolo_eje_y = -1;
		umbral_modulo = 30;
		cos_min = 0.05;
		cos_max = 0.95;
		aplicarSobelManual(imagen, grad_x, grad_y, Theta, Modulos, Modulos_dibujar, GX, GY);
		
	}
	else if (strcmp(argv[2], "-s") == 0) {
		simbolo_eje_y = -1;
		umbral_modulo = 30;
		cos_min = 0.05;
		cos_max = 0.95;
		aplicarSobelOpenCV(imagen, grad_x, grad_y, Theta, Modulos, Modulos_dibujar, GX, GY);
	}
	else if (strcmp(argv[2], "-c") == 0) {
		umbral_modulo = 65;
		cos_min = 0.10;
		cos_max = 0.90;
		aplicarCanny(imagen, grad_x, grad_y, abs_grad_x, abs_grad_y, Theta, Modulos, Modulos_dibujar, GX, GY);

	}
	
	cv::imshow("Contornos GX", GX);
	cv::imshow("Contornos GY", GY);
	cv::imshow("Contornos Theta", Theta/PI *128 );
	cv::imshow("Contornos Modulos", Modulos_dibujar);

	int numFilas = Modulos.size[0], numColumnas = Modulos.size[1];

	for (int i = 0; i < numFilas; ++i) {
		for (int j = 0; j < numColumnas; ++j) {
			Votos.at<unsigned char>(i, j) = 0;

			if (Modulos.at<double>(i, j) > umbral_modulo) {
				int x = (j - numColumnas / 2);
				int y = simbolo_eje_y * (numFilas / 2 - i);

				double angulo_positivo = Theta.at<double>(i, j);
				if (angulo_positivo < 0) angulo_positivo += 2 * PI;

				if ((abs(cos(angulo_positivo)) < cos_max) && (abs(cos(angulo_positivo)) > cos_min)) {
					double p = x*cos(angulo_positivo) + y*sin(angulo_positivo);
					Votos.at<unsigned char>(i, j) = 255;

					int eleccion = p / cos(angulo_positivo);
					if (((eleccion + numColumnas / 2) >= 0) && ((eleccion + numColumnas / 2) < numColumnas)) puntos_votacion[eleccion + numColumnas / 2]+=1;
				}
			}
		}
	}
	cv::imshow("Votos", Votos);

	int max_value = 0, best_index;
	for (int i = 0; i < imagen.size[1]; i++) {
		if (puntos_votacion[i] > max_value) {
			max_value = puntos_votacion[i];
			best_index = i;
		}
	}
	cvtColor(imagen, imagen, CV_GRAY2BGR);
	Point mejor(best_index, numFilas / 2);
	std::cout << "Punto de fuga: " << mejor << endl;
	cv::drawMarker(imagen, mejor, cv::Scalar(0, 0, 255), MARKER_CROSS, 15, 1);
	cv::imshow("Punto de fuga", imagen);


	std::cout << numFilas / 2 << " " << best_index << endl;
	
	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape

}

