#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <fstream>

using namespace cv;
using namespace std;

#define CHI_CUADRADA_4 9.49

struct Muestra
{
	double momento0;
	double momento1;
	double momento2;
	double perimetro;
	int label;
};
void umbralizar(Mat);
int labelToInt(std::string label);

inline bool exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer));
}



std::string intToLabel(int label)
{
	switch (label) {
	case 0:
		return "Es un triangulo";
	case 1:
		return "Es un rectangulo";
	case 2:
		return "Es un vagon";
	case 3:
		return "Es un rueda";
	case 4:
		return "Es un circulo";
	}
}

double distanciaMahalanobis(double valor, double media, double varianza) 
{
	return (valor - media)*(valor - media) / varianza;
}

double reconocido(char * nombre_fichero, vector<Point> contour) {
	ifstream source;                    // build a read-Stream
	source.open(nombre_fichero, ios_base::in);  // open data
	if (!source) {                     // if it does not work
		cerr << "Can't open Data!\n";
	}

	double m1, m2, m3, m4, v1, v2, v3, v4;
	Moments mu;
	mu = moments(contour, false);
	source >> m1;
	source >> m2;
	source >> m3;
	source >> m4;

	source >> v1;
	source >> v2;
	source >> v3;
	source >> v4;

	double acum = 0;
	acum += distanciaMahalanobis(mu.m00, m1, v1);
	acum += distanciaMahalanobis((mu.mu20 + mu.mu02), m2, v2);
	acum += distanciaMahalanobis((mu.mu20 + mu.mu02)*(mu.mu20 + mu.mu02) + 4 * mu.mu11*mu.mu11, m3, v3);
	acum += distanciaMahalanobis(arcLength(contour, true), m4, v4);

	return acum;
}
int main(int argc, char** argv)
{
	char * medias_triangulo = "./Triangulos medias.txt";
	char * medias_rectangulo = "./Rectangulos medias.txt";
	char * medias_vagon = "./Vagon medias.txt";
	char * medias_rueda = "./Ruedas medias.txt";
	char * medias_circulo = "./Circulos medias.txt";
	RNG rng(12345);
	int recorte = 5;

	Mat imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	imagen = imagen(cv::Rect(recorte, recorte, imagen.cols - recorte - 1, imagen.rows - recorte - 1));
	vector<vector<Point> > contours;
	umbralizar(imagen);
	findContours(imagen, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if (true) {
		/// Get the moments
		vector<Moments> mu(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mu[i] = moments(contours[i], false);
		}

		///  Get the mass centers:
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		/// Draw contours
		Mat drawing = Mat::zeros(imagen.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8);
			circle(drawing, mc[i], 4, color, -1, 8, 0);
		}

		/// Show in a window
		if (true) {
			imshow("Umbralizada", imagen);
			imshow("Contours", drawing);
			cout << "m0: " << mu[0].m00 << " m1: " << (mu[0].mu20 + mu[0].mu02) << " m2: " << (mu[0].mu20 + mu[0].mu02)*(mu[0].mu20 + mu[0].mu02) + 4 * mu[0].mu11*mu[0].mu11 << " perimetro: " << arcLength(contours[0], true) << endl;
			cout << "Numero de figuras: " << contours.size() << endl;
		}
	}
	
	double predict[5];
	int num_positivos = 0;
	
	for (int i = 0; i < contours.size(); i++) {
		int num_positivos = 0;
		cout << "Para la figura " << i+1 << ": " << endl;
		predict[0] = reconocido(medias_triangulo, contours[i]);
		predict[1] = reconocido(medias_rectangulo, contours[i]);
		predict[2] = reconocido(medias_vagon, contours[i]);
		predict[3] = reconocido(medias_rueda, contours[i]);
		predict[4] = reconocido(medias_circulo, contours[i]);

		int respuesta; double probabilidad = CHI_CUADRADA_4; int mejor;
		for (int i = 0; i < 5; i++) {
			if (predict[i] < CHI_CUADRADA_4) {
				num_positivos++;
				respuesta = i;
				if (predict[i] < probabilidad) {
					probabilidad = predict[i];
					mejor = i;
				}
			}
		}
		switch (num_positivos) {
		case 1:
			cout << intToLabel(respuesta);
			break;
		case 0:
			cout << "Objeto desconocido.";
			break;
		default:
			cout << "El sistema duda entre " << num_positivos << " objetos. (Mejor: " << intToLabel(mejor) << ")";
		}
		cout << endl;
		
	}
	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape

}
