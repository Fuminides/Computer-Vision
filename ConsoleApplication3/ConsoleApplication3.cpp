// Javier Fumanal Idocin, 684229
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

//Constantes
#define PROFUNDIDAD 5
#define TAMANO_FILTRO 5

using namespace cv;

//Funciones de efectos
Mat distorsionar(Mat , float);
void alien(Mat);
void poster(Mat, int);
void imageGhosting(Mat, double);
void equalizarGris(Mat);
void equalizarColor(Mat);
void mejoraConstraste(Mat, double, double);
double ** gauss(int);
Mat aplicar_filtro(Mat, double ** );
void on_trackbar0(int, void*);
void on_trackbar1(int, void*);
void on_trackbar2(int, void*);

//Efectos a activar.
const bool ECUALIZAR_GREY = false, ECUALIZAR_COLOR = false, ALIEN = true, POSTER = false, GHOSTING = false,
GAUSSIANO = false, DISTORSION = false, AUMENTAR_CONTRASTE = false;

Mat frameOriginal;
Mat frameAnterior[PROFUNDIDAD];

bool guarda = false;
int frames_acumulados = 0;
double tolerancias[] = { 0.3, 0.5, 0.2 };
int t1 = 0, t2 = 0, t3 = 0;

int main(int argc, char** argv)
{
	VideoCapture captura;
	double ** filtro;

	int omega = 2;
	double alfa = 0.5;
	int numero_colores = 230;
	double k1 = 256.0 / 10000000.0;
	double alpha = 0.5, beta = 1;

	// La 0 es la camara normal. 1 es la frontal
	// La camara frontal (Webcam) Esta rota en la Surface.

	if (!captura.open(0))
		return -1;

	namedWindow("Original", 0);
	namedWindow("Modificada", 1);
	if (GAUSSIANO) {
		filtro = gauss(omega);
	}
	if (ALIEN) {
		createTrackbar("R/G", "Modificada", &t1, 20, on_trackbar0);
		createTrackbar("R/B", "Modificada", &t2, 20, on_trackbar0);
		createTrackbar("G/B", "Modificada", &t3, 20, on_trackbar0);

		/*on_trackbar0(t1, 0);
		on_trackbar1(t2, 0);
		on_trackbar2(t3, 0)*/;
	}

	while (true)
	{
		Mat frame, frameOriginal;
		
		captura >> frame;
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
		if (AUMENTAR_CONTRASTE) {
			mejoraConstraste(frame, alpha, beta);
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
 * Si k > 0 -> Barril, Si k < 0 -> Cojin
 */
Mat distorsionar(Mat imagenOrigen, float k1) {
	Mat mapeoX, mapeoY, output;
	double ptoPrincipalX = imagenOrigen.rows/2,
		ptoPrincipalY = imagenOrigen.cols/2;

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
		frameAnterior[PROFUNDIDAD - 1] = (frameAnterior[PROFUNDIDAD - 1]*alfa / double(PROFUNDIDAD));
		for (int i = 0; i < PROFUNDIDAD - 1; i++) {
			frameAnterior[PROFUNDIDAD - 1] += frameAnterior[i]*alfa / double(PROFUNDIDAD);
		}
		frame = frame*(1 - alfa) + frameAnterior[PROFUNDIDAD - 1];
	}
	else {
		frames_acumulados++;
		if (frames_acumulados == PROFUNDIDAD) {
			guarda = true;
		}
	}

	for (int i = 0; i < PROFUNDIDAD - 1; i++) {
		frameAnterior[i + 1] = frameAnterior[i].clone();
	}
	frameAnterior[0] = frame.clone();
}

/**
 * Aplica el efecto "alien" a la imagen
 */
void alien(Mat frame) {
	double proporciones_piel[] = { 255.0 / 219.0, 255.0 / 172.0, 219.0 / 172.0 }; //R/G, R/B, G/B

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
				(abs(R - B) > 10) &&
				(abs(B - G) > 10) &&
				(abs(R - G) > 5)) {
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
	double ** coef =(double **) malloc(TAMANO_FILTRO*sizeof(double *));
	double sum = 0;

	for (int i = 0; i < TAMANO_FILTRO; i++) {
		coef[i] = (double *) malloc(TAMANO_FILTRO * sizeof(double));
		for (int j = 0; j < TAMANO_FILTRO; j++) {
			double termino = exp(-(i*i + j*j) / (2 * alfa*alfa));
			coef[i][j] = termino;
			sum = sum + termino;
		}
	}

	for (int i = 0; i < TAMANO_FILTRO; i++) {
		for (int j = 0; j < TAMANO_FILTRO; j++) {
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

			for (int x = 0; x < TAMANO_FILTRO; x++) {
				for (int y = 0; y < TAMANO_FILTRO; y++) {
					int indiceI = i + x - TAMANO_FILTRO / 2,
						indiceZ = z + y - TAMANO_FILTRO / 2;
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

/**
 * Aumenta contraste de la imagen.
 */
void mejoraConstraste(Mat frame, double alpha, double beta){
	std::vector<Mat> channels;

	cvtColor(frame, frame, CV_BGR2YCrCb);
	split(frame, channels);
	channels[0] = channels[0] * alpha + beta;
	merge(channels, frame);
	cvtColor(frame, frame, CV_YCrCb2BGR);
}

void on_trackbar0(int, void*)
{
	tolerancias[0] = ((double) t1) / 10.0;
}
void on_trackbar1(int, void*)
{
	tolerancias[1] = ((double)t2) / 10.0;
}
void on_trackbar2(int, void*)
{
	tolerancias[2] = ((double)t3) / 10.0;
}
