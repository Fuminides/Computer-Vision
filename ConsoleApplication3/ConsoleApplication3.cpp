// ConsoleApplication3.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

#define PROFUNDIDAD 5
#define n 5
using namespace cv;

Mat distorsionar(Mat , float);
void alien(Mat);
void poster(Mat, int);
void imageGhosting(Mat, double);
void equalizarGris(Mat);
void equalizarColor(Mat);
double ** gauss(int);
Mat aplicar_filtro(Mat, double ** );

const bool ECUALIZAR_GREY = false, ECUALIZAR_COLOR = true, ALIEN = false, POSTER = false, GHOSTING = false,
GAUSSIANO = true, DISTORSION = false;

Mat frameOriginal;
Mat frameAnterior[PROFUNDIDAD];
bool guarda = false;
int frames_acumulados = 0;
double alfa = 0.9;
int numero_colores = 230;
double k1 = -0.0000005;

int main(int argc, char** argv)
{
	VideoCapture cap;
	double ** filtro;
	int dim_filtro = 3, omega = 5;
	// La 0 es la camara normal. 1 es la frontal
	// La camara frontal (Webcam) Esta rota en la Surface.

	if (!cap.open(0))
		return 0;
	if (GAUSSIANO) {
		filtro = gauss(omega);
	}
	while (true)
	{
		Mat frame, frameOriginal;

		cap >> frame;
		frameOriginal = frame.clone();
		
		if (DISTORSION) {
			frame = distorsionar(frame, k1);
		}
		if (ALIEN) {
			alien(frame);
		}
		if (POSTER) {
			poster(frame, numero_colores);
		}
		if (GHOSTING) {
			imageGhosting(frame, alfa);
		}
		if (ECUALIZAR_GREY) {
			equalizarGris(frame);
		}
		if (ECUALIZAR_COLOR) {
			equalizarColor(frame);
		}
		if (GAUSSIANO) {
			frame = aplicar_filtro(frame, filtro);
		}
		
		if (frame.empty()) break; //Si algo falla, escapamos el bucle
		imshow("Modificada", frame);
		imshow("Original", frameOriginal);
		if (waitKey(10) == 27) break; //Para con la tecla escape
	}

	return 0;
}

/**
 * Aplica una distorsion a la imagen dada.
 * Si k > 0 -> Cojin, Si k < 0 -> Barril
 */
Mat distorsionar(Mat imagenOrigen, float k1) {
	Mat mapeoX, mapeoY, output;
	double ptoPrincipalX = imagenOrigen.rows,
		ptoPrincipalY = imagenOrigen.cols;

	mapeoX.create(imagenOrigen.size(), CV_32FC1);
	mapeoY.create(imagenOrigen.size(), CV_32FC1);
	int rows = imagenOrigen.rows, cols = imagenOrigen.cols;

	for (int i = 0; i < rows; i++) {
		for (int z = 0; z < cols; z++) {
			double r_cuadrado = pow(i - ptoPrincipalX, 2) + pow(z - ptoPrincipalY,2);
		
			mapeoX.at<float>(i, z) = (z - ptoPrincipalY) * (1+ k1 * r_cuadrado) + ptoPrincipalY;
			mapeoY.at<float>(i, z) = (i - ptoPrincipalX) * (1 + k1 * r_cuadrado) + ptoPrincipalX;
		}
	}

	remap(imagenOrigen, output, mapeoX, mapeoY, CV_INTER_LINEAR);

	return output;
}

/**
 * Equaliza la imagen en un canal de grises. (Pone tambien la imagen original en un solo canal).
 */
void equalizarGris(Mat frame) {
	cvtColor(frame, frame, CV_BGR2GRAY);
	frameOriginal = frame.clone();
	equalizeHist(frame, frame);
}

/**
* Equaliza la imagen en un canal de colores.
*/
void equalizarColor(Mat frame) {
	std::vector<Mat> channels;

	cvtColor(frame, frame, CV_BGR2YCrCb);
	split(frame, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, frame);
	cvtColor(frame, frame, CV_YCrCb2BGR);
}

/**
 * Aplica un efecto de ghosting a la imagen.
 */
void imageGhosting(Mat frame, double alfa) {
	if (guarda) {
		frameAnterior[PROFUNDIDAD - 1] = frameAnterior[PROFUNDIDAD - 1] / double(PROFUNDIDAD);
		for (int i = 0; i < PROFUNDIDAD - 1; i++) {
			frameAnterior[PROFUNDIDAD - 1] += frameAnterior[i] / double(PROFUNDIDAD);
		}
		frame = frame*(1 - alfa) + frameAnterior[PROFUNDIDAD - 1] * alfa;
	}
	else {
		frames_acumulados++;
		if (frames_acumulados == PROFUNDIDAD) {
			guarda = true;
		}
	}

	for (int i = 0; i < PROFUNDIDAD - 1; i++) {
		frameAnterior[i + 1] = frameAnterior[i];
	}
	frameAnterior[0] = frame.clone();
}

/**
 * Aplica el efecto "alien" a la imagen
 */
void alien(Mat frame) {
	double proporciones_piel[] = { 255.0 / 219.0, 255.0 / 172.0, 219.0 / 172.0 }; //R/G, R/B, G/B
	double tolerancias[] = { 0.3, 0.7, 0.2 };

	for (int i = 0; i < frame.rows; i++) {
		for (int z = 0; z < frame.cols; z++) {
			Vec3b canales = frame.at<Vec3b>(i, z);
			unsigned char G = canales[1], R = canales[2], B = canales[0];

			double proporcionRG = R*1.0 / (G*1.0), proporcionRB = R*1.0 / (B*1.0),
				proporcionGB = G*1.0 / (B*1.0);
			//std::cout << "Valores: " << std::to_string(proporcionRG) << " " << std::to_string(abs(proporcionRG - proporciones_piel[1])) << std::endl;
			if ((abs(proporcionRG - proporciones_piel[0]) < tolerancias[0]) &&
				(abs(proporcionRB - proporciones_piel[1]) < tolerancias[1]) &&
				(abs(proporcionGB - proporciones_piel[2]) < tolerancias[2]) &&
				(abs(R - B) > 10)) {
				//std::cout << "Entro" << std::endl;
				canales[0] = 0;
				canales[1] = 0;
				canales[2] = 255;
			}
			frame.at<Vec3b>(i, z) = canales;
		}
	}
}

/**
 * Devuelve un array con los coefecientes de un filtro guassiano nxn con omega como parametro.
 */
double ** gauss(int alfa) {
	double ** coef =(double **) malloc(n*sizeof(double *));
	double sum = 0;

	for (int i = 0; i < n; i++) {
		coef[i] = (double *) malloc(n * sizeof(double));
		for (int j = 0; j < n; j++) {
			double termino = exp(-(i*i + j*j) / (2 * alfa*alfa));
			coef[i][j] = termino;
			sum = sum + termino;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			coef[i][j] = coef[i][j] / sum;
		}
	}

	return coef;
}
/**
 * Reduce el numero de colores en la imagen al numero especificado.
 */
void poster(Mat frame, int numero_colores) {
	for (int i = 0; i < frame.rows; i++) {
		for (int z = 0; z < frame.cols; z++) {
			Vec3b canales = frame.at<Vec3b>(i, z);
			unsigned char G = canales[1], R = canales[2], B = canales[0];

			canales[0] = B % numero_colores *1.0 / numero_colores * 255;
			canales[1] = G % numero_colores *1.0 / numero_colores * 255;
			canales[2] = R % numero_colores *1.0 / numero_colores * 255;
			frame.at<Vec3b>(i, z) = canales;
		}
	}
}

/**
 * Aplica un filtro a una imagen.
 */
Mat aplicar_filtro(Mat frame, double ** filtro) {
	int rows = frame.rows, cols = frame.cols;
	Mat resultado = frame.clone();

	for (int i = 0; i < rows; i++) {
		for (int z = 0; z < cols; z++) {
			Vec3b color(0, 0, 0); //Mejorable. Como coger el tipo de la matriz?

			for (int x = 0; x < n; x++) {
				for (int y = 0; y < n; y++) {
					int indiceI = i + x - n / 2,
						indiceZ = z + y - n / 2;
					if ((indiceI > 0) && (indiceI < rows) && (indiceZ > 0) && (indiceZ < cols)) {
						//std::cout << "Intentamos: x:" << indiceI << " , y: " << indiceZ << std::endl;
						color = color + filtro[x][y]
							*frame.at<Vec3b>(indiceI,indiceZ);
					}
				}
			}
			resultado.at<Vec3b>(i,z) = color;
		}
	}

	return resultado;

}