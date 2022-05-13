from tkinter import *
import tkinter as tk
import tkinter.ttk as CB
from tkinter import messagebox
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import pandas as pd
import main
from recommenders.datasets import movielens

MOVIELENS_DATA_SIZE = '100k'
# def getNameMovie():
data_item_URL = 'ml-100k/u.item'
data_item = pd.read_table(data_item_URL,
                          sep='|', header=None, encoding="windows-1251")
data_item.columns = ['itemID', 'nameMovie', 'year', 'none', 'url', 'Action', 'Adventure', 'Animation', "Children's",
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']

print('lan 0')
# print(str(data_item.sample(15)))
# print(type(data_item))
movieResult = pd.DataFrame(data_item, columns=['itemID', 'nameMovie', 'year'])
a = movieResult.nameMovie
# pandas.core.frame.DataFrame
NameMovie = a.to_numpy()
DataMovie = NameMovie.tolist();
print(len(DataMovie))

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    genres_col='genre',
    header=["userID", "itemID", "rating"]
)


def getId_Item():
    n = combb1.get();
    iduser = inputIduser.get(1.0, "end-1c")
    for i in range(0,len(DataMovie)):
        if(DataMovie[i] == n):
            # print("id_item",i+1)
            # print("iduder",iduser)
            return i
    return i

def Recommender():
    iduser = inputIduser.get(1.0, "end-1c")
    # iduser = 2
    if(len(iduser) < 1):
        messagebox.showinfo("Recommender System", "Vui lòng nhập id của bạn !")
    elif(iduser.isnumeric() == False):
        messagebox.showinfo("Recommender System", "Vui lòng nhập đúng id của bạn !")
    else:
        itemId = getId_Item()+1
        # print("itemId",getId_Item()+1)
        # print("iduser",iduser)
        resultRecommender = main.test(int(iduser),itemId,data)
        main.sample_recommendation([int(iduser)])

        Result = resultRecommender.tolist();
        if(len(Result) > 0):
            result1.config(text= Result[0][1])
            result2.config(text=Result[1][1])
            result3.config(text=Result[2][1])
            result4.config(text=Result[3][1])
            result5.config(text=Result[4][1],fg= "black",bg= 'white' ,font= "Time 10 bold")
            result6.config(text=Result[5][1])
            result7.config(text=Result[6][1])
            result8.config(text=Result[7][1])
            result9.config(text=Result[8][1])
            result10.config(text=Result[9][1])

            resultCollumName.place(x=5, y=10)
            result1.place(x=5, y=45)
            result2.place(x=5, y=65)
            result3.place(x=5, y=85)
            result4.place(x=5, y=105)
            result5.place(x=5, y=125)
            result6.place(x=5, y=145)
            result7.place(x=5, y=165)
            result8.place(x=5, y=185)
            result9.place(x=5, y=205)
            result10.place(x=5, y=225)

            print(resultRecommender.tolist())
        else:
            print("có lỗi rồi")



app = Tk()
app.title("VKU.UDN.VN-KHOA CÔNG NGHỆ THÔNG TIN - TRUYỀN THÔNG ĐẠI HỌC VIỆT-HÀN")
app.geometry('1180x580+100+20')
app.resizable(False, False)
app.config(bg="white")
la = Label(app, background="white", width="1135", height=630).pack()
image2 =Image.open('logoVKU.png')
reder = ImageTk.PhotoImage(image2)
img = Label(app, image= reder,bg="white").place(x=280, y=0)
# Ten nhom
labelframe=LabelFrame(app,text="Sinh viên thực hiện",fg = '#F7961E',font= "Time 8 bold",width=280,
             height=100,highlightcolor="red",bg="white",
             highlightbackground="white",highlightthickness=1)
labelframe.place(x = 860, y = 5)
# lableframe item
labtextinput = Label(labelframe,text=" Giảng viên hướng dẫn: ",fg= "red",bg= 'white', font= "Time 8 bold")
labtextinput.place(x=0,y=5)
labtexNameGV = Label(labelframe,text="TS. Nguyễn Sĩ Thìn ",fg= "red",bg= 'white' ,font= "Time 8 bold")
labtexNameGV.place(x=130,y=5)

labtexHoTen = Label(labelframe,text=" Họ tên sinh viên: ",fg= "red",bg= 'white', font= "Time 8 bold")
labtexHoTen.place(x=0,y=24)
labtexHVT = Label(labelframe,text="Hồ Văn Trung ",fg= "red",bg= 'white' ,font= "Time 8 bold")
labtexHVT.place(x=130,y=24)
labtexNVC = Label(labelframe,text="Nguyễn Văn Chiến ",fg= "red",bg= 'white' ,font= "Time 8 bold")
labtexNVC.place(x=130,y=40)
labtexLop = Label(labelframe,text=" Lớp sinh hoạt: ",fg= "red",bg= 'white', font= "Time 8 bold")
labtexLop.place(x=0,y=57)
labtexLSH = Label(labelframe,text="18IT4 ",fg= "red",bg= 'white' ,font= "Time 8 bold")
labtexLSH.place(x=130,y=57)

# TOPIC
labtexTopic = Label(app,text="BÁO CÁO ĐỒ ÁN MÔN HỌC CHUYÊN ĐỀ 6",fg= "black",bg= 'white' ,font= "Time 18 bold")
labtexTopic.place(x=300,y=90)
# LightFM/Hybrid Matrix Factorization
labtexNameTopic = Label(app,text="LIGHTFM/HYBRID MATRIX FACTORIZATION CHO MOVILENS",fg= "black",bg= 'white' ,font= "Time 18 bold")
labtexNameTopic.place(x=200,y=125)

# Content
labelframeContent=LabelFrame(app,text="Recommender system",fg = 'red',font= "Time 8 bold",width=1000,
             height=380,highlightcolor="red",bg="white",
             highlightbackground="red",highlightthickness=1)
labelframeContent.place(x = 80, y = 180)

# enter iduser and itemid
labelframeEnter=LabelFrame(labelframeContent,text="Nhập đầu vào cho hệ thống",fg = '#FBBC05',font= "Time 8 bold",width=300,
             height=340,highlightcolor="yellow",bg="white",
             highlightbackground="yellow",highlightthickness=1)
labelframeEnter.place(x = 10, y = 10)

# choose iduser
labtexNameIDuser = Label(labelframeEnter,text="Nhập idUser của bạn:",fg= "red",bg= 'white' ,font= "Time 10 bold")
labtexNameIDuser.place(x=5,y=10)
inputIduser = tk.Text(labelframeEnter,height = 1, width = 32)
inputIduser.place(x=5, y = 35)
# choose the film name

labtexNameNameFilm = Label(labelframeEnter,text="Chọn bộ phim muốn xem :",fg= "red",bg= 'white' ,font= "Time 10 bold")
labtexNameNameFilm.place(x=5,y=70)
combb1 = CB.Combobox(labelframeEnter, width = 35, height = 9, font = "Time 10 bold",state = "readonly")
combb1['values'] = (DataMovie)
combb1.current(1)
combb1.place(x= 5, y = 100)
# button recommender
btn =Button(labelframeEnter, text = "Gợi ý phim", font = "Time 10 bold", bg = "green", fg = "white", command =Recommender)
btn.place(x= 110, y= 150)

#result recommender
labelframeOutput=LabelFrame(labelframeContent,text="Kết quả hệ thống gợi ý bộ phim liên quan",fg = 'blue',font= "Time 8 bold",width=630,
             height=340,highlightcolor="yellow",bg="white",
             highlightbackground="blue",highlightthickness=1)
labelframeOutput.place(x = 340, y = 10)
resultCollumName = Label(labelframeOutput,text="Tên bộ phim ",fg= "red",bg= 'white' ,font= "Time 10 bold")
# resultCollumName.place(x=5,y=10)
# resultCollumnYear = Label(labelframeOutput,text="Năm phát hành ",fg= "black",bg= 'white' ,font= "Time 10 bold")
# resultCollumnYear.place(x=150,y=10)
# resultCollumnUrl = Label(labelframeOutput,text="URL",fg= "black",bg= 'white' ,font= "Time 10 bold")
# resultCollumnUrl.place(x=250,y=10)
result1 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result2 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result3 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result4 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result5 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result6 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result7 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result8 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result9= Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")
result10 = Label(labelframeOutput,text="Kết quả gợi ý",fg= "black",bg= 'white' ,font= "Time 10 bold")

# result1.place(x=5,y=10)

#Tạo menu
# menubar = Menu(app)
# app.config(menu=menubar)
app.mainloop()