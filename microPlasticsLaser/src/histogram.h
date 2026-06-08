// histogramWindow.h
// Real-time momentum histogram visualization for microplastics simulation

#ifndef HISTOGRAM_WINDOW_H
#define HISTOGRAM_WINDOW_H

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Configuration
// ============================================================================
#define HIST_NUM_BINS 50
#define HIST_WINDOW_WIDTH 1200
#define HIST_WINDOW_HEIGHT 400

// ============================================================================
// Histogram Data Structures
// ============================================================================
typedef struct {
    int bins[HIST_NUM_BINS];
    float minVal;
    float maxVal;
    float binWidth;
    int maxCount;
    char label[32];
    float colorR, colorG, colorB;
} Histogram;

// Global histogram variables
int HistogramWindow = -1;
Histogram HistPx, HistPy, HistPz;
int HistogramInitialized = 0;

float MomentumRangeMin = -1.0f;
float MomentumRangeMax = 1.0f;
int HistogramAutoScale = 1;

FILE* HistMovieFile = NULL;
int* HistBuffer = NULL;
int HistMovieFlag = 0;

// ============================================================================
// Function Prototypes
// ============================================================================
void initHistogram(Histogram* hist, const char* label, float r, float g, float b);
void clearHistogram(Histogram* hist);
void addToHistogram(Histogram* hist, float value);
void finalizeHistogram(Histogram* hist);
void drawHistogram(Histogram* hist, float x, float y, float width, float height);
void drawHistogramLabels(Histogram* hist, float x, float y, float width, float height);
void renderBitmapString(float x, float y, void* font, const char* string);
void histogramDisplay(void);
void histogramReshape(int w, int h);
void createHistogramWindow(int mainArgc, char** mainArgv, int numMicroplastics);
void updateMomentumHistograms(float4* bodyVelocity, float4* bodyForce,
                              int numberOfPolymers, int numberOfBodies);
void histogramMovieOn(void);
void histogramMovieOff(void);

// ============================================================================
// Histogram Functions
// ============================================================================
void initHistogram(Histogram* hist, const char* label, float r, float g, float b)
{
    memset(hist->bins, 0, sizeof(hist->bins));
    hist->minVal = MomentumRangeMin;
    hist->maxVal = MomentumRangeMax;
    hist->binWidth = (hist->maxVal - hist->minVal) / HIST_NUM_BINS;
    hist->maxCount = 0;
    strncpy(hist->label, label, 31);
    hist->label[31] = '\0';
    hist->colorR = r;
    hist->colorG = g;
    hist->colorB = b;
}

void clearHistogram(Histogram* hist)
{
    memset(hist->bins, 0, sizeof(hist->bins));
    hist->maxCount = 0;
}

void addToHistogram(Histogram* hist, float value)
{
    if (value < hist->minVal) value = hist->minVal;
    if (value > hist->maxVal) value = hist->maxVal;

    int binIndex = (int)((value - hist->minVal) / hist->binWidth);

    if (binIndex < 0) binIndex = 0;
    if (binIndex >= HIST_NUM_BINS) binIndex = HIST_NUM_BINS - 1;

    hist->bins[binIndex]++;
}

void finalizeHistogram(Histogram* hist)
{
    hist->maxCount = 0;
    for (int i = 0; i < HIST_NUM_BINS; i++) {
        if (hist->bins[i] > hist->maxCount) {
            hist->maxCount = hist->bins[i];
        }
    }
    if (hist->maxCount == 0) hist->maxCount = 1;
}

void renderBitmapString(float x, float y, void* font, const char* string)
{
    glRasterPos2f(x, y);
    while (*string) {
        glutBitmapCharacter(font, *string);
        string++;
    }
}

// ============================================================================
// Draw Histogram (Bell Curve Style)
// ============================================================================
void drawHistogram(Histogram* hist, float x, float y, float width, float height)
{
    float barWidth = width / HIST_NUM_BINS;
    float barHeightScale = height / (float)hist->maxCount;

    // Background
    glColor3f(0.15f, 0.15f, 0.15f);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();

    // Grid lines
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINES);
    for (int i = 0; i <= 4; i++) {
        float yLine = y + (height * i / 4.0f);
        glVertex2f(x, yLine);
        glVertex2f(x + width, yLine);
    }
    glEnd();

    // Zero momentum line
    float zeroX = x + width * (0.0f - hist->minVal) / (hist->maxVal - hist->minVal);
    if (zeroX >= x && zeroX <= x + width) {
        glColor3f(0.5f, 0.5f, 0.5f);
        glBegin(GL_LINES);
        glVertex2f(zeroX, y);
        glVertex2f(zeroX, y + height);
        glEnd();
    }

    // Histogram bars (all going UP from bottom)
    for (int i = 0; i < HIST_NUM_BINS; i++) {
        float barHeight = hist->bins[i] * barHeightScale;
        float barX = x + i * barWidth;

        float intensity = (float)hist->bins[i] / (float)hist->maxCount;
        glColor3f(hist->colorR * (0.3f + 0.7f * intensity),
                  hist->colorG * (0.3f + 0.7f * intensity),
                  hist->colorB * (0.3f + 0.7f * intensity));

        glBegin(GL_QUADS);
        glVertex2f(barX + 1, y);
        glVertex2f(barX + barWidth - 1, y);
        glVertex2f(barX + barWidth - 1, y + barHeight);
        glVertex2f(barX + 1, y + barHeight);
        glEnd();
    }

    // Border
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();
}

// ============================================================================
// Draw Labels with Units
// ============================================================================
void drawHistogramLabels(Histogram* hist, float x, float y, float width, float height)
{
    char buffer[64];

    // Title
    glColor3f(hist->colorR, hist->colorG, hist->colorB);
    renderBitmapString(x + width/2 - 30, y + height + 20,
                      GLUT_BITMAP_HELVETICA_18, hist->label);

    // X-axis label
    glColor3f(0.9f, 0.9f, 0.9f);
    renderBitmapString(x + width/2 - 50, y - 35,
                      GLUT_BITMAP_HELVETICA_10, "Momentum (kg*m/s)");

    // X-axis values
    glColor3f(0.8f, 0.8f, 0.8f);
    sprintf(buffer, "%.2e", hist->minVal);
    renderBitmapString(x, y - 20, GLUT_BITMAP_HELVETICA_10, buffer);

    sprintf(buffer, "%.2e", (hist->minVal + hist->maxVal) / 2.0f);
    renderBitmapString(x + width/2 - 20, y - 20, GLUT_BITMAP_HELVETICA_10, buffer);

    sprintf(buffer, "%.2e", hist->maxVal);
    renderBitmapString(x + width - 40, y - 20, GLUT_BITMAP_HELVETICA_10, buffer);

    // Y-axis label (stacked)
    glColor3f(0.9f, 0.9f, 0.9f);
    renderBitmapString(x - 45, y + height/2 + 30, GLUT_BITMAP_HELVETICA_10, "P");
    renderBitmapString(x - 45, y + height/2 + 20, GLUT_BITMAP_HELVETICA_10, "a");
    renderBitmapString(x - 45, y + height/2 + 10, GLUT_BITMAP_HELVETICA_10, "r");
    renderBitmapString(x - 45, y + height/2,      GLUT_BITMAP_HELVETICA_10, "t");
    renderBitmapString(x - 45, y + height/2 - 10, GLUT_BITMAP_HELVETICA_10, "i");
    renderBitmapString(x - 45, y + height/2 - 20, GLUT_BITMAP_HELVETICA_10, "c");
    renderBitmapString(x - 45, y + height/2 - 30, GLUT_BITMAP_HELVETICA_10, "l");
    renderBitmapString(x - 45, y + height/2 - 40, GLUT_BITMAP_HELVETICA_10, "e");
    renderBitmapString(x - 45, y + height/2 - 50, GLUT_BITMAP_HELVETICA_10, "s");

    // Y-axis values
    glColor3f(0.8f, 0.8f, 0.8f);
    sprintf(buffer, "0");
    renderBitmapString(x - 20, y, GLUT_BITMAP_HELVETICA_10, buffer);

    sprintf(buffer, "%d", hist->maxCount / 2);
    renderBitmapString(x - 25, y + height/2, GLUT_BITMAP_HELVETICA_10, buffer);

    sprintf(buffer, "%d", hist->maxCount);
    renderBitmapString(x - 25, y + height - 5, GLUT_BITMAP_HELVETICA_10, buffer);
}

// ============================================================================
// Display Callback
// ============================================================================
void histogramDisplay(void)
{
    glutSetWindow(HistogramWindow);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, HIST_WINDOW_WIDTH, 0, HIST_WINDOW_HEIGHT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float margin = 60.0f;
    float topMargin = 50.0f;
    float bottomMargin = 55.0f;
    float spacing = 30.0f;

    float totalWidth = HIST_WINDOW_WIDTH - 2 * margin - 2 * spacing;
    float histWidth = totalWidth / 3.0f;
    float histHeight = HIST_WINDOW_HEIGHT - topMargin - bottomMargin;

    glColor3f(1.0f, 1.0f, 1.0f);
    renderBitmapString(HIST_WINDOW_WIDTH/2 - 100, HIST_WINDOW_HEIGHT - 25,
                      GLUT_BITMAP_HELVETICA_18, "Microplastic Momentum Distribution");

    float x1 = margin;
    drawHistogram(&HistPx, x1, bottomMargin, histWidth, histHeight);
    drawHistogramLabels(&HistPx, x1, bottomMargin, histWidth, histHeight);

    float x2 = margin + histWidth + spacing;
    drawHistogram(&HistPy, x2, bottomMargin, histWidth, histHeight);
    drawHistogramLabels(&HistPy, x2, bottomMargin, histWidth, histHeight);

    float x3 = margin + 2 * (histWidth + spacing);
    drawHistogram(&HistPz, x3, bottomMargin, histWidth, histHeight);
    drawHistogramLabels(&HistPz, x3, bottomMargin, histWidth, histHeight);

    glutSwapBuffers();

    if (HistMovieFlag == 1 && HistBuffer != NULL && HistMovieFile != NULL) {
        glReadPixels(0, 0, HIST_WINDOW_WIDTH, HIST_WINDOW_HEIGHT,
                    GL_RGBA, GL_UNSIGNED_BYTE, HistBuffer);
        fwrite(HistBuffer, sizeof(int) * HIST_WINDOW_WIDTH * HIST_WINDOW_HEIGHT,
               1, HistMovieFile);
    }
}

void histogramReshape(int w, int h)
{
    glutSetWindow(HistogramWindow);
    glViewport(0, 0, w, h);
}

// ============================================================================
// Create Window
// ============================================================================
void createHistogramWindow(int mainArgc, char** mainArgv, int numMicroplastics)
{
    if (HistogramInitialized) return;

    initHistogram(&HistPx, "Px (X Momentum)", 1.0f, 0.3f, 0.3f);
    initHistogram(&HistPy, "Py (Y Momentum)", 0.3f, 1.0f, 0.3f);
    initHistogram(&HistPz, "Pz (Z Momentum)", 0.3f, 0.3f, 1.0f);

    HistogramWindow = glutCreateWindow("Momentum Histograms");
    glutPositionWindow(100, 100);
    glutReshapeWindow(HIST_WINDOW_WIDTH, HIST_WINDOW_HEIGHT);

    glutDisplayFunc(histogramDisplay);
    glutReshapeFunc(histogramReshape);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    HistBuffer = (int*)malloc(HIST_WINDOW_WIDTH * HIST_WINDOW_HEIGHT * sizeof(int));

    HistogramInitialized = 1;
    printf("\n Histogram window created.\n");
}

// ============================================================================
// Update Histograms
// ============================================================================
void updateMomentumHistograms(float4* bodyVelocity, float4* bodyForce,
                              int numberOfPolymers, int numberOfBodies)
{
    if (!HistogramInitialized || HistogramWindow < 0) return;

    if (HistogramAutoScale) {
        float minP = 1e30f, maxP = -1e30f;

        for (int i = numberOfPolymers; i < numberOfBodies; i++) {
            float mass = bodyForce[i].w;
            float px = mass * bodyVelocity[i].x;
            float py = mass * bodyVelocity[i].y;
            float pz = mass * bodyVelocity[i].z;

            if (px < minP) minP = px;
            if (py < minP) minP = py;
            if (pz < minP) minP = pz;
            if (px > maxP) maxP = px;
            if (py > maxP) maxP = py;
            if (pz > maxP) maxP = pz;
        }

        float range = maxP - minP;
        if (range < 1e-10f) range = 1.0f;

        MomentumRangeMin = minP - 0.1f * range;
        MomentumRangeMax = maxP + 0.1f * range;

        HistPx.minVal = HistPy.minVal = HistPz.minVal = MomentumRangeMin;
        HistPx.maxVal = HistPy.maxVal = HistPz.maxVal = MomentumRangeMax;
        HistPx.binWidth = HistPy.binWidth = HistPz.binWidth = 
            (MomentumRangeMax - MomentumRangeMin) / HIST_NUM_BINS;
    }

    clearHistogram(&HistPx);
    clearHistogram(&HistPy);
    clearHistogram(&HistPz);

    for (int i = numberOfPolymers; i < numberOfBodies; i++) {
        float mass = bodyForce[i].w;
        float px = mass * bodyVelocity[i].x;
        float py = mass * bodyVelocity[i].y;
        float pz = mass * bodyVelocity[i].z;

        addToHistogram(&HistPx, px);
        addToHistogram(&HistPy, py);
        addToHistogram(&HistPz, pz);
    }

    finalizeHistogram(&HistPx);
    finalizeHistogram(&HistPy);
    finalizeHistogram(&HistPz);

    glutSetWindow(HistogramWindow);
    glutPostRedisplay();
}

// ============================================================================
// Video Recording
// ============================================================================
void histogramMovieOn(void)
{
    if (HistMovieFile == NULL) {
        HistMovieFile = fopen("histogram_movie.raw", "wb");
        if (HistMovieFile != NULL) {
            HistMovieFlag = 1;
            printf("Histogram movie recording started.\n");
        }
    }
}

void histogramMovieOff(void)
{
    if (HistMovieFile != NULL) {
        fclose(HistMovieFile);
        HistMovieFile = NULL;
        HistMovieFlag = 0;
        printf("Histogram movie recording stopped.\n");
    }
}

#endif // HISTOGRAM_WINDOW_H

//git test 