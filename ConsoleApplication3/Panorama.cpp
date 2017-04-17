#include "stdafx.h"
#include <cstdio>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define USE_BRISK 0
#define USE_ORB 1
#define USE_FAST 2

#define USE_BRUTE 0
#define USE_VECINO 1
#define USE_FLANN 2

vector<KeyPoint> compute_keypoints(Mat imagen, int mode) {
	Ptr<FeatureDetector> detector;
	switch (mode) {
	case USE_FAST:
		detector = FastFeatureDetector::create(40);
		break;
	case USE_BRISK:
		detector = BRISK::create();
		break;
	}

	vector<KeyPoint> keypoints;
	detector->detect(imagen, keypoints);

	return keypoints;
}

Mat extraer_descriptores(Mat imagen, vector<KeyPoint> keypoints, int mode) {
	Ptr<DescriptorExtractor> descriptorExtractor;
	switch (mode) {
	case USE_BRISK:
		descriptorExtractor = BRISK::create();
		break;
	case USE_ORB:
		descriptorExtractor = ORB::create();
		break;
	}
	Mat descriptors;

	descriptorExtractor->compute(imagen, keypoints, descriptors);

	return descriptors;
}

vector<DMatch> match_descriptors(Mat descriptors1, Mat descriptors2, int mode) {
	vector<DMatch> matches;
	if (mode == USE_BRUTE){
		Ptr<BFMatcher> matcher = BFMatcher::create();
		matcher->match(descriptors1, descriptors2, matches);

		std::vector< DMatch > good_matches;
		double distancia_minima = INFINITY;
		for (int i = 0; i < descriptors1.rows; i++)
		{
			if (matches[i].distance < distancia_minima)
			{
				distancia_minima = matches[i].distance;
			}
		}
		for (int i = 0; i < descriptors1.rows; i++)
		{
			if ((matches[i].distance <= 3 * distancia_minima) | (matches[i].distance <= 100))
			{
				good_matches.push_back(matches[i]);
			}
		}
		matches = good_matches;
	}
	else if (mode == USE_ORB) {
		for (int i = 0; i < descriptors1.size[0]; i++) {
			double mejor_distancia = INFINITY, segunda_mejor = INFINITY;
			int posX, posY;
			for (int z = 0; z < descriptors2.size[0]; z++) {
				Mat distancias_cuadradas = (descriptors1.row(i) - descriptors2.row(z)).mul(descriptors1.row(i) - descriptors2.row(z));
				double distancia = sum(distancias_cuadradas)[0];
				if (distancia < mejor_distancia) {
					mejor_distancia = distancia;
					posX = i;
					posY = z;
				}
				else if (distancia < segunda_mejor) {
					segunda_mejor = distancia;
				}
			}
			if (mejor_distancia < 0.8*segunda_mejor) {
				DMatch match(posY, posX,0,mejor_distancia);
				matches.push_back(match);
			}
		}
	}
	else if (mode == USE_FAST) {
		if (descriptors1.type() != CV_32F) {
			descriptors1.convertTo(descriptors1, CV_32F);
		}
		if (descriptors2.type() != CV_32F) {
			descriptors2.convertTo(descriptors2, CV_32F);
		}
		Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
		matcher->match(descriptors1, descriptors2, matches);

		std::vector< DMatch > good_matches;
		double distancia_minima = INFINITY;
		for (int i = 0; i < descriptors1.rows; i++)
		{
			if (matches[i].distance < distancia_minima)
			{
				distancia_minima = matches[i].distance;
			}
		}
		for (int i = 0; i < descriptors1.rows; i++)
		{
			if (matches[i].distance <= 3 * distancia_minima)
			{
				good_matches.push_back(matches[i]);
			}
		}
		matches = good_matches;
	}

	for(int i = 0; i < matches.size(); i++) cout << matches[i].distance << " " << matches[i].imgIdx << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
	return matches;
}


void matching_disco(int argc, char ** argv) {
	Mat imagen1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat imagen2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	vector<KeyPoint> k1, k2;

	k1 = compute_keypoints(imagen1, USE_BRISK);
	k2 = compute_keypoints(imagen2, USE_BRISK);
	Mat descriptores1 = extraer_descriptores(imagen1, k1, USE_BRISK);
	Mat descriptores2 = extraer_descriptores(imagen2, k2, USE_BRISK);

	vector<DMatch> matches = match_descriptors(descriptores1, descriptores2, USE_BRUTE);

	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(imagen1, k1, imagen2, k2, matches, img_matches);
	imshow("matches", img_matches);
	waitKey(0);

}

void main(int argc, char ** argv) {
	matching_disco(argc, argv);
}