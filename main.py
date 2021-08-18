from run_adaboost import run_adaboost
from run_ID3 import run_ID3
from main_naiveBayes import run_naive_bayes
import os


# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    os.chdir("aclImdb")

    while True:
        print("\nARTIFICIAL INTELLIGENCE - PROJECT 2")
        method = input("INSERT TRAINING METHOD(1, 2, OR 3) TO IMPLEMENT\n"
              "1 : NAIVE BAYES\n"
              "2 : ID3\n"
              "3 : ADABOOST\n"
              "Select : ")

        if method == '1' :

            run_naive_bayes()

        elif method == '2':

            run_ID3()

        elif method == '3':

            run_adaboost()

        else:
            print("WRONG ANSWER!\n")

        repeat = input("\nDo you want to try some other algorithm?\n"
                       "0 : No\n"
                       "1 : Yes\n"
                       "Select : ")
        if repeat == '1':
            continue
        else :
            print("\nProgram exits...\n")
            break
