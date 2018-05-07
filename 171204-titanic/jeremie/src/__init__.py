import src.titanicdataanalysis.titanicdataanalysis as tda

choice ="null"

if __name__ == '__main__':

    while choice != 'e' and choice != 'exit':
        choice = input("| Welcome \n"
                       "|------------------------------- \n"
                       "| What's your choice ? \n"
                       "| 1 : First naive Titanic kaggle : with Random Forest and grid search\n"
                       "|------------------------------- \n"
                       "| Sortir : e \n")

        if choice == '1':
            # firstStep.exec(root_path ='..', log_level='INFO')
            returnCode = tda.loadingData(dataPath="../data/", logLevel='INFO')
            if returnCode !=1: print("something went bad ... return code : %d",returnCode)
            else: print("ok!")
        elif choice == '2':
            print("Soon ... ")
        # elif choice == '3':
            # thirdStep.exec(root_path ='..', log_level='INFO')
            # print("Hello !!!!")
        elif choice == 'e' or "exit":
            print('Bye Bye !')
        else:
            print('Bad choice. Do it again.\n')