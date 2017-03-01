// Javier Fumanal Idocin, 684229
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

//Constantes
#define PROFUNDIDAD 5
#define TAMANO_FILTRO 4

using namespace cv;

//Funciones de efectos
Mat distorsionar(Mat , double, double);
void alien(Mat);
void poster(Mat, int);
void imageGhosting(Mat, double);
void equalizarGris(Mat);
void equalizarColor(Mat);
void comic(Mat, double, double);
void aumentarConstraste(Mat, double, double);
double ** gauss(int);
Mat aplicar_filtro(Mat, double ** );
void histograma(Mat, std::string);
void histogramaGris(Mat,std::string);
void on_trackbar0(int, void*);
void on_trackbar1(int, void*);
void on_trackbar2(int, void*);
void on_trackbar3(int, void*);
void on_trackbar4(int, void*);
void on_trackbar5(int, void*);

//Efectos a activar.
const bool ECUALIZAR_GREY = false, ECUALIZAR_COLOR = false, ALIEN = true, POSTER = false, GHOSTING = false,
GAUSSIANO = false, DISTORSION = false, AUMENTAR_CONTRASTE = false, COMIC = false;

Mat frameOriginal;
Mat frameAnterior[PROFUNDIDAD];

bool guarda = false;
int frames_acumulados = 0;
double tolerancias[] = { 0.3, 0.5, 0.2 };
int t1 = 0, t2 = 0, t3 = 0, modulo = 0, modulo2 = 0;
int positivo = 0;
double k1 = 0.0, k2 = 0.0;


int main(int argc, char** argv)
{
	VideoCapture captura;
	double ** filtro;

	int omega = 5;
	double alfa = 0.8;
	int numero_colores = 16;
	double alpha = 0.25, beta = 20;

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
		createTrackbar("R/B", "Modificada", &t2, 20, on_trackbar1);
		createTrackbar("G/B", "Modificada", &t3, 20, on_trackbar2);
	}
	if (DISTORSION) {
		createTrackbar("K1: ", "Modificada", &modulo, 5, on_trackbar3);
		createTrackbar("K2: ", "Modificada", &modulo2, 5, on_trackbar4);
		createTrackbar("Signo: ", "Modificada", &positivo, 1, on_trackbar5);
	}

	while (true)
	{
		Mat frame, frameOriginal;
		
		captura >> frame;
		frameOriginal = frame.clone();
		
		if (DISTORSION) {
			frame = distorsionar(frame, k1, k2);
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
		if (COMIC) {
			comic(frame, alpha, beta);
		}
		if (AUMENTAR_CONTRASTE) {
			aumentarConstraste(frame, alpha, beta);
		}
		
		if (frame.empty()) break; //Si algo falla, escapamos el bucle
		if (!ECUALIZAR_GREY) {
			imshow("Modificada", frame);
			histograma(frame, "Histograma Modificada");
			imshow("Original", frameOriginal);
			histograma(frameOriginal, "Histograma Original");
		}

		if (waitKey(10) == 27) break; //Para con la tecla escape
	}

	return 0;
}

/**
 * Aplica una distorsion a la imagen dada.
 * Si k > 0 -> Barril, Si k < 0 -> Cojin
 */
Mat distorsionar(Mat imagenOrigen, double k1, double k2) {
	Mat mapeoX, mapeoY, output;
	double ptoPrincipalX = imagenOrigen.rows/2,
		ptoPrincipalY = imagenOrigen.cols/2;

	mapeoX.create(imagenOrigen.size(), CV_32FC1);
	mapeoY.create(imagenOrigen.size(), CV_32FC1);
	int rows = imagenOrigen.rows, cols = imagenOrigen.cols;

	for (int i = 0; i < rows; i++) {
		for (int z = 0; z < cols; z++) {
			double r_cuadrado = pow(i - ptoPrincipalX, 2) + pow(z - ptoPrincipalY,2);
		
			mapeoX.at<float>(i, z) = (z - ptoPrincipalY) * (1+ k1 * r_cuadrado + k2 * r_cuadrado * r_cuadrado) + ptoPrincipalY;
			mapeoY.at<float>(i, z) = (i - ptoPrincipalX) * (1 + k1 * r_cuadrado + k2 * r_cuadrado * r_cuadrado) + ptoPrincipalX;
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
	histogramaGris(frame, "Frame modificado");
	histogramaGris(frameOriginal, "Frame original");
	imshow("Modificada", frame);
	imshow("Original", frameOriginal);
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
	double proporciones_piel[] = { 234.0 / 192.0, 234.0 / 134.0, 192.0 / 134.0 }; //R/G, R/B, G/B

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
 * Codigo sacado de: http://answers.opencv.org/question/27808/how-can-you-use-k-means-clustering-to-posterize-an-image-using-c/
 */
void poster(Mat src, int numero_colores) {
	/*
	for (int i = 0; i < frame.rows; i++) {
		for (int z = 0; z < frame.cols; z++) {
			Vec3b canales = frame.at<Vec3b>(i, z);
			unsigned char G = canales[1], R = canales[2], B = canales[0];

			canales[0] = B % numero_colores *1.0 / numero_colores * 255;
			canales[1] = G % numero_colores *1.0 / numero_colores * 255;
			canales[2] = R % numero_colores *1.0 / numero_colores * 255;
			frame.at<Vec3b>(i, z) = canales;
		}
	}*/
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y, x)[z];
	int clusterCount = numero_colores;
	Mat labels;
	int attempts = 2;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.01), attempts, KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			src.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			src.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			src.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
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
void comic(Mat frame, double alpha, double beta){
	std::vector<Mat> channels;
	Mat brillantes, brillantes_float;

	cvtColor(frame, frame, CV_BGR2YCrCb);
	split(frame, channels);
	threshold(channels[0], brillantes, beta, 1, 0);
	for (int i = 0; i < brillantes.size[0]; i++) {
		for (int z = 0; z < brillantes.size[1]; z++) {
			char resultado = channels[0].at<char>(i, z) + channels[0].at<char>(i, z)*(brillantes.at<char>(i,z) * 2 - 1)*alpha;
			if (resultado != 0) channels[0].at<char>(i, z) = resultado;
		}
	}
	merge(channels, frame);
	cvtColor(frame, frame, CV_YCrCb2BGR);
}
void aumentarConstraste(Mat frame, double alpha, double beta) {
	std::vector<Mat> channels;
	cvtColor(frame, frame, CV_BGR2YCrCb);
	split(frame, channels);
	channels[0] = channels[0] * alpha + beta;
	merge(channels, frame);
	cvtColor(frame, frame, CV_YCrCb2BGR);
}
/**
 * Muestra el histograma por pantalla de la imagen dada.
 */
void histograma(Mat src, std::string nombre) {
	std::vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow(nombre, CV_WINDOW_AUTOSIZE);
	imshow(nombre, histImage);

}

/**
* Muestra el histograma por pantalla de la imagen dada.
*/
void histogramaGris(Mat src, std::string nombre) {
	std::vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	/// Display
	namedWindow(nombre, CV_WINDOW_AUTOSIZE);
	imshow(nombre, histImage);

}
/**
 * Funciones de trackbar para las tolerancias del color
 */
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

/**
 * Trackbar para la distorsion
 */
void on_trackbar3(int, void*) //Modulo de k1
{
	k1 = ((k1>=0)-(k1<0)) * (double(modulo) / 1000000.0);
	std::cout << "K1: " << k1 << " K2: " << k2 << std::endl;
}
void on_trackbar4(int, void*) //Modulo de k2
{
	k2 = ((k2>=0) - (k2<0)) * (double(modulo2) / 1000000.0);
	std::cout << "K1: " << k1 << " K2: " << k2 << std::endl;
}
void on_trackbar5(int, void*) //Signo de las k
{
	if (positivo == 0) {
		k1 = abs(k1);
		k2 = abs(k2);
	}
	else {
		k1 = -abs(k1);
		k2 = -abs(k2);
	}
	std::cout << "K1: " << k1 << " K2: " << k2 << std::endl;
}
