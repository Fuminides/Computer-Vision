#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <fstream>

using namespace cv;
using namespace std;

#define CHI_CUADRADA_4 13.277
#define REDUCIR_SESGO 2.0

//Declaracion de funciones externas
void umbralizar(Mat);

//FUNCIONES

/**
 * Indica a que figura corresponde el int pasado por parametro.
 */
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
		return "Es una rueda";
	case 4:
		return "Es un circulo";
	}
}

/**
 * Devuelve la distancia de mahalanobis de un valor, dado su media y varianza.
 */
double distanciaMahalanobis(double valor, double media, double varianza) 
{
	return (valor - media)*(valor - media) / (varianza+(varianza*REDUCIR_SESGO)+REDUCIR_SESGO);
}

/**
 * Devuelve el valor de la distancia de mahalanobis total de un objeto con respecto a una clase.
 */
double reconocido(char * nombre_fichero, double area, double hu[10], double perimetro) {
	ifstream source;                    
	source.open(nombre_fichero, ios_base::in); 
	if (!source) {                     
		cerr << "Ruta incorrecta, revisar fichero.\n";
	}

	double m1, m2, m3, m4, v1, v2, v3, v4;
	
	source >> m1;
	source >> m2;
	source >> m3;
	source >> m4;

	source >> v1;
	source >> v2;
	source >> v3;
	source >> v4;

	double acum = 0;
	acum += distanciaMahalanobis(area, m1, v1);
	acum += distanciaMahalanobis(hu[0], m2, v2);
	acum += distanciaMahalanobis(hu[1], m3, v3);
	acum += distanciaMahalanobis(perimetro, m4, v4);

	return acum;
}

int mainClasificar(int argc, char** argv)
{
	char * medias_triangulo = "./Triangulos medias.txt";
	char * medias_rectangulo = "./Rectangulos medias.txt";
	char * medias_vagon = "./Vagon medias.txt";
	char * medias_rueda = "./Ruedas medias.txt";
	char * medias_circulo = "./Circulos medias.txt";
	int recorte = 5;
	int exitos = 1;
	vector<vector<Point> > contours;


	Mat imagen = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	//A veces los bordes se cargan negros, así que los eliminamos recortando los bordes de la imagen.
	imagen = imagen(cv::Rect(recorte, recorte, imagen.cols - recorte - 1, imagen.rows - recorte - 1));
	umbralizar(imagen);
	findContours(imagen, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	
	

	//Clasificamos cada figura encontrada:
	for (int i = 0; i < contours.size(); i++) {
		int num_positivos = 0, respuesta, mejor;
		double probabilidad = CHI_CUADRADA_4;
		double predict[5];
		Moments mu = moments(contours[i], true);
		double hu[10];
		HuMoments(mu, hu);
		double perimetro = arcLength(contours[i], true);
		if (perimetro > 7) { //Filtro por si algun punto espureo no deseado ha sido detectado como contorno.
			cout << "Para la figura " << exitos << ": (Posicion: " << Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00) << ")" << endl;
			exitos++;

			predict[0] = reconocido(medias_triangulo, mu.m00, hu, perimetro);
			predict[1] = reconocido(medias_rectangulo, mu.m00, hu, perimetro);
			predict[2] = reconocido(medias_vagon, mu.m00, hu, perimetro);
			predict[3] = reconocido(medias_rueda, mu.m00, hu, perimetro);
			predict[4] = reconocido(medias_circulo, mu.m00, hu, perimetro);
			cout << predict[0] << endl;
			cout << predict[1] << endl;

			cout << predict[2] << endl;
			cout << predict[3] << endl;
			cout << predict[4] << endl;

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
	}
	while (true)
		if (waitKey(10) == 27) break; //Para con la tecla escape

}
