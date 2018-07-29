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
#define PARENT(i) ((i-1)/2)
#define LEFT(i) (2*i+1)
#define RIGHT(i) (2*i+2)

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

// /** Newton iteration sqrt algorithm.
//  */
// float fast_sqrt(float a) {
//   float x = a, y = 0.;
//   while (fabs(x - y) > 0.001) {
//     y = x;
//     x = 0.5 * (x + a / x);
//   }
//   return x;
// }

/** Aluxiliary swap function.
 */
void swap(float* pa, float* pb) {
  float temp = *pa;
  *pa = *pb;
  *pb = temp;
}

static int counter = 0;

/** KNN_removal the depth pixels in a depth image.
 */
void KNN_removal(const float *imgIn, float *imgOut, const unsigned int &width, const unsigned int &height,
                 const unsigned int &kernelSize, const unsigned int k, const float distance_thr) {
  const int halfSize = kernelSize / 2;

  // Set a inner small kernel to fasten caclulation.
  const int halfSqrtK = sqrt(k) / 2 + 1;

  const float angle_width_per_pixel = M_PI * 2. / width;
  const float angle_height_per_pixel = M_PI * 2. / height;

  // Go through all pixels.
  float heap[k];
  float center_depth;
  float depth_squared;
  float width_per_pixel;
  float height_per_pixel;

  #pragma omp parallel for reduction(+:counter) private(heap, center_depth, depth_squared, width_per_pixel, height_per_pixel)
  for (int n = halfSize; n < height - halfSize; ++n) {

    for (int m = halfSize; m < width - halfSize; ++m) {
      if (imgIn[m + n * width] != 0.f) {

       /* KNN_removeValue START. */
        // Parameters.
        center_depth = imgIn[m + n * width];
        depth_squared = center_depth * center_depth;
        width_per_pixel = sqrt(2 * depth_squared - cos(angle_width_per_pixel) * 2 * depth_squared);
        height_per_pixel = sqrt(2 * depth_squared - cos(angle_height_per_pixel) * 2 * depth_squared);

        // Loop through the kernel.
        float distance, heap_max;
        int size = 0;

        // Loop through the inner small kernel.
        //   Smallest k elems are mostly likely to be all inside this small kernel.
        for (int x = halfSize - halfSqrtK; x < halfSize + halfSqrtK; ++x) {
          for (int y = halfSize - halfSqrtK; y < halfSize + halfSqrtK; ++y) {

            distance = fabs(imgIn[m - halfSize + x + (n - halfSize + y) * width] - center_depth)
                     + width_per_pixel * abs(x - halfSize) + height_per_pixel * abs(y - halfSize);
            if (distance != 0.f) {

             /* neighbors_insert START. */
              if (size < k) {
                heap[size] = distance;
                int i = size;
                size++;
                while (i != 0 && heap[i] > heap[PARENT(i)]) {
                  swap(&heap[i], &heap[PARENT(i)]);
                  i = PARENT(i);
                }
                heap_max = heap[0];
              } else if (distance < heap_max) {
                heap[0] = distance;
                int i = 0;
                while ((RIGHT(i) < k && (heap[i] < heap[LEFT(i)] || heap[i] < heap[RIGHT(i)]))
                    || (RIGHT(i) >= k && LEFT(i) < k && heap[i] < heap[LEFT(i)])) {
                  if (RIGHT(i) < k) {
                    if (heap[LEFT(i)] > heap[RIGHT(i)]) {
                      swap(&heap[i], &heap[LEFT(i)]);
                      i = LEFT(i);
                    } else {
                      swap(&heap[i], &heap[RIGHT(i)]);
                      i = RIGHT(i);
                    }
                  } else {
                    swap(&heap[i], &heap[LEFT(i)]);
                    i = LEFT(i);
                  }
                }
                heap_max = heap[0];
              }
             /* neighbors_insert END. */

            }
          }
        }

        // Scan through outer kernel, add in exceptions.
        for (int x = 0; x < kernelSize; ++x) {
          for (int y = 0; y < kernelSize; ++y) {
            if (x >= halfSize - halfSqrtK && x < halfSize + halfSqrtK &&
                y >= halfSize - halfSqrtK && y < halfSize + halfSqrtK)
              continue;

            distance = fabs(imgIn[m - halfSize + x + (n - halfSize + y) * width] - center_depth)
                     + width_per_pixel * abs(x - halfSize) + height_per_pixel * abs(y - halfSize);
            if (distance != 0.f) {

             /* neighbors_insert START. */
              if (size < k) {
                heap[size] = distance;
                int i = size;
                size++;
                while (i != 0 && heap[i] > heap[PARENT(i)]) {
                  swap(&heap[i], &heap[PARENT(i)]);
                  i = PARENT(i);
                }
                heap_max = heap[0];
              } else if (distance < heap_max) {
                heap[0] = distance;
                int i = 0;
                while ((RIGHT(i) < k && (heap[i] < heap[LEFT(i)] || heap[i] < heap[RIGHT(i)]))
                    || (RIGHT(i) >= k && LEFT(i) < k && heap[i] < heap[LEFT(i)])) {
                  if (RIGHT(i) < k) {
                    if (heap[LEFT(i)] > heap[RIGHT(i)]) {
                      swap(&heap[i], &heap[LEFT(i)]);
                      i = LEFT(i);
                    } else {
                      swap(&heap[i], &heap[RIGHT(i)]);
                      i = RIGHT(i);
                    }
                  } else {
                    swap(&heap[i], &heap[LEFT(i)]);
                    i = LEFT(i);
                  }
                }
                heap_max = heap[0];
              }
             /* neighbors_insert END. */

            }
          }
        }

        // Get the k nearest vaild neighbors.
        float k_sum = 0.f;
        for (int i = 0; i < k; i++)
          k_sum += heap[i];
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
