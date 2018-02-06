#ifndef QUICKSORT_H
#define QUICKSORT_H

#include <vector>
#include <cstdlib>
#include <time.h>

int rand(int left, int right)
{
    srand((unsigned)time(NULL));
    int randNum = rand() % (right - left) + left;
    return randNum;
}

template <typename T>
int partition(std::vector<T>& inpArr, int left, int right, bool (*compFunc)(T, T))
{
    int pivotPos = rand(left, right);
    T pivot = inpArr[pivotPos];
    inpArr[pivotPos] = inpArr[left];
    inpArr[left] = pivot;
    int i = left + 1;
    int j = right;

    while (i <= j){
        while (compFunc(inpArr[i],pivot) && i < right)
            i++;
        while (!compFunc(inpArr[j],pivot) && j >= left + 1)
            j--;
        if (i < j){
            T tmp = inpArr[i];
            inpArr[i] = inpArr[j];
            inpArr[j] = tmp;
        }
        else
            break;
    }
    inpArr[left] = inpArr[j];
    inpArr[j] = pivot;
    return j;
}

template <typename T>
void quicksort(std::vector<T>& inpArr, int left, int right, bool (*compFunc)(T, T ))
{
    if (left < right){
        int pivotPos = partition(inpArr, left, right, compFunc);
        if (pivotPos == left)
            quicksort(inpArr, left + 1, right, compFunc);
        else if (pivotPos == right)
            quicksort(inpArr, left, right - 1, compFunc);
        else {
            quicksort(inpArr, left, pivotPos - 1, compFunc);
            quicksort(inpArr, pivotPos + 1, right, compFunc);
        }
    }
}

#endif // QUICKSORT_H

