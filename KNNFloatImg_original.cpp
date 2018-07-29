#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <math.h>
#include <chrono>
#include <omp.h>

#define INF 0xFFFF

using namespace std;

void usage(){
  cerr << "Usage: KNNFloatImg inputImage outputImage k  distance_thr kernelSize(should be 511 but"
          "you can use 21 to debug)" << endl;
}

/** Reads an image for the file inFile. Creates an array of the corect size in the heap.
 *  Returns true on success.
 */
bool readImage(float *&image, const char *inFile, unsigned int &width, unsigned int &height) {
  ifstream bIn;
  // Fix the size of the two dimension variables to 32 bit.
  uint32_t width32 = 0;
  uint32_t height32 = 0;
  // Open the file for binary reading.
  bIn.open(inFile, std::ofstream::in | std::ofstream::binary);
  if (!bIn.is_open()) {
    cerr << "Error opening file " << inFile << " for reading!" << endl;
    return false;
  }
  // Read the width and height.
  bIn.read( (char *)&width32, sizeof(width32));
  bIn.read( (char *)&height32, sizeof(height32));
  // Error checking.
  if(width32 == 0 || height32 == 0) {
    cerr << "Width or height 0" << endl;
    return false;
  }
  // Setup the image.
  width = width32;
  height = height32;
  image = new float[width * height];
  cout << "Reading width " << width << " height " << height << " = " << width * height << endl;
  // Prepare reading the image.
  float *imgInPtr = image;
  float *imgEndPtr = image + width * height;
  // Read the input image.
  while(bIn && imgInPtr != imgEndPtr) {
    // Actually read the data. Using a block copy might be faster - but it is quite fast anyways.
    bIn.read((char *)imgInPtr++, sizeof(float));
  }
  return true;
}


/** Writes the float array img to outFile (as binary floats, first 32bit width and height
 *  unsinged_integers).
 */
bool writeImage(const float *img, const char *outFile, const unsigned int width, const unsigned int height) {
  ofstream bOut;
  // Open the file for writing.
  bOut.open(outFile,  std::ofstream::binary);
  if (!bOut.is_open()) {
    cerr << "Error opening file " << outFile << " for writing!" << endl;
    return false;
  }
  // Write the width and height.
  uint32_t width32 = width;
  uint32_t height32 = height;
  bOut.write((char *)&width32, sizeof(width32));
  bOut.write((char *)&height32, sizeof(height32));
  // Write the data.
  const float *imgPtr = img;
  const float *endPtr = img + (width * height);
  while(imgPtr != endPtr)
    bOut.write((const char *)imgPtr++, sizeof(float));
  return true;
}

/** Newton iteration sqrt algorithm.
 */
float fast_sqrt(float a) {
  float x = a, y = 0.;
  while (fabs(x - y) > 0.001) {
    y = x;
    x = 0.5 * (x + a / x);
  }
  return x;
}

static int counter = 0;

/** KNN_removal the depth pixels in a depth image.
 */
void KNN_removal(const float *imgIn, float *imgOut, const unsigned int &width, const unsigned int &height,
                 const unsigned int &kernelSize, const unsigned int k, const float distance_thr) {
  const int halfSize = kernelSize / 2;

  const float reverse_cos_angle_width_per_pixel = 1 - cos(M_PI * 2. / width);
  const float reverse_cos_angle_height_per_pixel = 1 - cos(M_PI * 2. / height);

  // Go through all pixels.
  #pragma omp parallel for reduction(+:counter)
  for (int n = halfSize; n < height - halfSize; ++n) {

    // Neighbors.
    float neighbors_dist[k];
    int neighbors_prev[k];
    int neighbors_next[k];
    int head, tail;
    float neighbors_max;

    for (int m = halfSize; m < width - halfSize; ++m) {
      if (imgIn[m + n * width] != 0.f) {

       /* KNN_removeValue START. */
        // Initilaize neighbors.
        for (int i = 0; i < k; i++) { 
          neighbors_dist[i] = INF;
          neighbors_next[i] = i + 1;
          neighbors_prev[i] = i - 1;
        }
        head = 0;
        tail = k - 1;
        neighbors_max = INF;

        // Parameters.
        const float center_depth = imgIn[m + n * width];
        const float depth_squared = center_depth * center_depth;
        const float width_per_pixel = fast_sqrt(2 * depth_squared * reverse_cos_angle_width_per_pixel);
        const float height_per_pixel = fast_sqrt(2 * depth_squared * reverse_cos_angle_height_per_pixel);

        // Loop through the kernel.
        float distance;
        for (int x = 0; x < kernelSize; ++x) {
          for (int y = 0; y < kernelSize; ++y) {

            distance = fabs(imgIn[m - halfSize + x + (n - halfSize + y) * width] - center_depth)
                     + width_per_pixel * abs(x - halfSize) + height_per_pixel * abs(y - halfSize);
            if (distance != 0.f && distance < neighbors_max) {

             /* neighbors_insert START. */
              neighbors_dist[tail] = distance;
              int i = head;
              while (distance > neighbors_dist[i])
                i = neighbors_next[i];
              if (i == head) {
                neighbors_next[tail] = head;
                neighbors_prev[head] = tail;
                head = tail;
                tail = neighbors_prev[tail];
                neighbors_next[neighbors_prev[head]] = k;
                neighbors_prev[head] = -1;
              } else if (i != tail) {
                int tmp_tail = tail;
                neighbors_next[neighbors_prev[tail]] = k;
                neighbors_next[tail] = i;
                neighbors_next[neighbors_prev[i]] = tail;
                tail = neighbors_prev[tail];
                neighbors_prev[tmp_tail] = neighbors_prev[i];
                neighbors_prev[i] = tmp_tail;
              }
              neighbors_max = neighbors_dist[tail];
             /* neighbors_insert END. */

            }
          }
        }

        // Get the k nearest vaild neighbors.
        float k_sum = 0.f;
        for (int i = 0; i < k; ++i)
          k_sum += neighbors_dist[i];
        if (k_sum / k > distance_thr) {
          counter++;
          imgOut[m + n * width] = 0.f;
        } else 
          imgOut[m + n * width] = imgIn[m + n * width];
       /* KNN_removeValue END. */

      } else
        imgOut[m + n * width] = 0.f;
    }
  }
}

/** The main function. Five steps:
 *    1) check the program arguments.
 *    2) create the Kernel.
 *    3) read the input image.
 *    4) KNN_removal.
 *    5) write the output image.
 */
int main(int argc, char* argv[])
{
  // 1) Check the program arguments.
  if(argc != 6) {
    usage();
    return 0;
  }
  unsigned int kernelSize = 511;  // This 511 is what the kernel should really be.
  kernelSize = atoi(argv[5]);
  unsigned int width = 0;         // Image dimensions.
  unsigned int height = 0;
  unsigned int k = 0;
  float distance_thr = 0;
  k = atoi(argv[3]);
  distance_thr = atof(argv[4]);
  // 2) Create the kernel.
  if(kernelSize % 2 == 0) {
    cout << "Kernel size must be odd!" << endl;
    return 0;
  }
  float *imgIn = 0;
  // 3) Read the image.
  if(!readImage(imgIn, argv[1], width, height))
    return 0;
  float *imgOut = new float[width * height];
  std::chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
  // 4) KNN_removal the image.
  KNN_removal(imgIn, imgOut, width, height, kernelSize, k, distance_thr);
  std::chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
  unsigned int microsecs = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
  cout << " Micro seconds: " << microsecs << endl;
  cout << "Outliers: " << counter << endl;
  // 5) Write the output.
  writeImage(imgOut, argv[2], width, height);
  return 0;
}
