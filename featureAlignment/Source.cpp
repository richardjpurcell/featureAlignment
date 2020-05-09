/*
 * File:	featureAlignment/Source.cpp
 * Author:	Richard Purcell
 * Date:	2020/05/04
 * Version	1.0
 *
 * Purpose: A program that aligns images based on features
 *          Done for OpenCV's Computer Vision 1 course
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat img;

int main(int argc, char** argv) {

	string filename;

	if (argc != 2)
	{
		cout << "Usage: featureAlignment.exe filename" << endl;
		filename = "./emir.jpg";
	}
	else
	{
		filename = argv[1];
	}

	img = imread(filename, IMREAD_GRAYSCALE);
	Size sz = img.size();
	int height = sz.height / 3;
	int width = sz.width;

	vector<Mat>channels;
	channels.push_back(img(Rect(0, 0, width, height)));
	channels.push_back(img(Rect(0, height, width, height)));
	channels.push_back(img(Rect(0, 2 * height, width, height)));

	Mat blue = channels[0];
	Mat green = channels[1];
	Mat red = channels[2];

	imshow("blue", blue);
	imshow("green", green);

	int MAX_FEATURES = 1000;
	float GOOD_MATCH_PERCENT = 0.1f;

	vector<KeyPoint> keypointsBlue, keypointsGreen, keypointsRed;
	Mat descriptorsBlue, descriptorsGreen, descriptorsRed;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(blue, Mat(), keypointsBlue, descriptorsBlue);
	orb->detectAndCompute(green, Mat(), keypointsGreen, descriptorsGreen);
	orb->detectAndCompute(red, Mat(), keypointsRed, descriptorsRed);

	// Match features
	std::vector<DMatch> matchesBlueGreen;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	matcher->match(descriptorsBlue, descriptorsGreen, matchesBlueGreen, Mat());

	// Sort matches by score
	sort(matchesBlueGreen.begin(), matchesBlueGreen.end());

	// Remove not so good matches
	int numGoodMatches = matchesBlueGreen.size() * GOOD_MATCH_PERCENT;
	matchesBlueGreen.erase(matchesBlueGreen.begin() + numGoodMatches, matchesBlueGreen.end());

	// Draw top matches
	Mat imMatchesBlueGreen;
	drawMatches(blue, keypointsBlue, green, keypointsGreen, matchesBlueGreen, imMatchesBlueGreen);
	imshow("Matches", imMatchesBlueGreen);
	
	// Match features between Red and Green channels
	vector<DMatch> matchesRedGreen;
	matcher->match(descriptorsRed, descriptorsGreen, matchesRedGreen, Mat());

	sort(matchesRedGreen.begin(), matchesRedGreen.end());

	numGoodMatches = matchesRedGreen.size() * GOOD_MATCH_PERCENT;
	matchesRedGreen.erase(matchesRedGreen.begin() + numGoodMatches, matchesRedGreen.end());

	Mat imMatchesRedGreen;
	drawMatches(red, keypointsRed, green, keypointsGreen, matchesRedGreen, imMatchesRedGreen);
	imshow("Matches Red", imMatchesRedGreen);
	// Extract location of good matches
	vector<Point2f> pointsBlue, pointsGreen;

	for (size_t i = 0; i < matchesBlueGreen.size(); i++)
	{
		pointsBlue.push_back(keypointsBlue[matchesBlueGreen[i].queryIdx].pt);
		pointsGreen.push_back(keypointsGreen[matchesBlueGreen[i].trainIdx].pt);
	}

	// Find homography
	Mat hBlue = findHomography(pointsBlue, pointsGreen, RANSAC);
	
	vector<Point2f> pointsRed, pointsGreen2;
	for (size_t i = 0; i < matchesRedGreen.size(); i++)
	{
		pointsRed.push_back(keypointsRed[matchesRedGreen[i].queryIdx].pt);
		pointsGreen2.push_back(keypointsGreen[matchesRedGreen[i].trainIdx].pt);
	}
	
	Mat hRed = findHomography(pointsRed, pointsGreen2, RANSAC); 
	
	// Use homography to warp image
	Mat blueWarped, redWarped;
	
	//cout << hRed.size() << endl;
	cout << hBlue.size() << endl;
	warpPerspective(blue, blueWarped, hBlue, green.size());
	warpPerspective(red, redWarped, hRed, green.size());

	//Merge Channels
	Mat colorImage;
	vector<Mat> colorImageChannels{ blueWarped, green, redWarped };
	merge(colorImageChannels, colorImage);
	
	//imshow("Blue Green Matches", blueWarped);
	//imshow("Red Green Matches", redWarped); 
	imshow("Final Out", colorImage);
	waitKey(0);

	destroyAllWindows();

	return 0;

}