
import os
def read_files_in_dir(file_dir_old,file_dir_new):  
    file_index=-1
    if os.path.exists(file_dir_old):
        for file in os.listdir(file_dir_old):
                file_index=file_index+1
                file_path = os.path.join(file_dir_old,file)    
                file_suffix=os.path.splitext(file)[-1]
                new_file_path=file_dir_new+str(file_index)+file_suffix
                os.rename(file_path,new_file_path)


if __name__=="__main__":
        file_dir_old="./cabliImage/"
        file_dir_new="./cabliImageNew/"
        read_files_in_dir(file_dir_old,file_dir_new)


