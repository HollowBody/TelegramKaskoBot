import logging

from telegram import Bot
from telegram import Update
from telegram import PhotoSize
from telegram import InlineKeyboardButton
from telegram import InlineKeyboardMarkup
from telegram.ext import CallbackContext
from telegram.ext import CallbackQueryHandler
from telegram.ext import Updater
from telegram.ext import MessageHandler
from telegram.ext import Filters
from telegram.utils.request import Request

import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

# система логгирования
logger = logging.getLogger(__name__)

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# main menu burttons
GET_OFFER = 'offer'
GET_PRODUCT_INFO = 'info'
GET_MANUAL = 'manual'

# maker menu buttons
GET_AUDI_MODELS = 'audi'
GET_BMW_MODELS = 'bmw'
GET_MB_MODELS = 'mb'

# confirmation menu bittons
CONFIRM = 'CONFIRM'
CANCEL = 'CANCEL'

# ключи 
isGetOfferClick = False
isGetBmwClick = False
isModelNotEmpty = False
isMileageNotEmpty = False
isManufactureYearNotEmpty = False
isEnginePowerNotEmpty = False
isFirstMessageSend = False

# файлы для работы модели
result_file_name = 'result.joblib.pkl'
scaler_file_name = 'scaler.joblib.pkl'
ohe_file_name = 'ohe.joblib.pkl'
LabelEncoder_file_name = 'labelEncoder.joblib.pkl'

scaler_loaded = load(open(scaler_file_name, 'rb'))
ohe_loaded = load(open(ohe_file_name, 'rb'))
LabelEncoder_loaded = load(open(LabelEncoder_file_name, "rb"))
result_loaded = load(open(result_file_name, "rb"))


# example params
mileage = 121999.5
engine_power = 100
manufacture_year = 2003
maker = 'bmw'
model = '330xd'

def predict(mileage: float, engine_power: int, manufacture_year: int, maker: str, model: str):
    # creating test data frame
    testDF = pd.DataFrame({"mileage":[mileage],"engine_power": [engine_power],"manufacture_year": [manufacture_year],"maker": [maker],"model": [model]})
    
    # devide on two data frame (categorical and decimal)
    testDecimal = testDF[['mileage','engine_power','manufacture_year']]
    testCategorical = testDF[['maker','model']].astype(str)
    
    # transform it!
    testCategoricalTransformed = LabelEncoder_loaded.transform(testCategorical)
    
    # train decimal on scaler
    testDecimalTransformed = scaler_loaded.transform(testDecimal)

    # train cat on OHE
    ohe_loaded.transform(testCategoricalTransformed)
    oheTestCategoricalTransformed = ohe_loaded.transform(testCategoricalTransformed)

    # stack cat and decimal data
    totalClassification = np.hstack((oheTestCategoricalTransformed,testDecimalTransformed))

    # get total normalized dataframe
    X = totalClassification
    
    # prediction
    Y = result_loaded.predict(X)  

    # choose programs and link
    if Y == 1:
        link = 'lite'
    elif Y == 2:
        link = 'Lux'
    elif Y == 0:
        link = 'Optimum'
    return link

# main menu
def get_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Оформить каско', callback_data=GET_OFFER),],
            [InlineKeyboardButton(text='Информация о продукте', callback_data=GET_PRODUCT_INFO),],
            [InlineKeyboardButton(text='Инструкция', callback_data=GET_MANUAL),],
            ],)

# maker menu
def get_maker_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Audi', callback_data=GET_AUDI_MODELS),],
            [InlineKeyboardButton(text='BMW', callback_data=GET_BMW_MODELS),],
            [InlineKeyboardButton(text='MB', callback_data=GET_MB_MODELS),],
            ],)

# confirmation menu
def get_confirmation_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Подтвердить', callback_data=CONFIRM),],
            [InlineKeyboardButton(text='Отказаться', callback_data=CANCEL),],
            ],)


# Message from user handler 
def message_handler(update: Update, context: CallbackContext):
    global isMileageNotEmpty
    global isEnginePowerNotEmpty
    global isManufactureYearNotEmpty
    global isModelNotEmpty 
    global isFirstMessageSend

    # get message text
    message_text = update.message.text
    if not(isFirstMessageSend):
        isFirstMessageSend = True
        return greeting_handler(update, context)

    # не заполнен пробег
    if isGetOfferClick and isGetBmwClick and isModelNotEmpty and isManufactureYearNotEmpty and isEnginePowerNotEmpty and not(isMileageNotEmpty):
        global mileage        
        mileage = float(message_text)
        isMileageNotEmpty = True
        reply_text = f'Нажмите \"Подтвердить\", если согласны на обработку данных и генерацию готового предложения. Для отмены нажмите \"Отказаться\"'
        update.message.reply_text(text=reply_text,reply_markup=get_confirmation_keyboard())
    # не заполнена мощность
    if isGetOfferClick and isGetBmwClick and isModelNotEmpty and isManufactureYearNotEmpty and not(isEnginePowerNotEmpty):
        global engine_power
        engine_power = int(message_text)       
        isEnginePowerNotEmpty = True
        reply_text = f'Введите показания одометра (пробег) автомобиля'
        update.message.reply_text(text=reply_text)
    # не заполнен год выпуска 
    if isGetOfferClick and isGetBmwClick and isModelNotEmpty and not(isManufactureYearNotEmpty):
        global manufacture_year 
        manufacture_year = int(message_text)        
        isManufactureYearNotEmpty = True
        reply_text = f'Введите мощность двигателя автомобиля'
        update.message.reply_text(text=reply_text)
    # не заполнена модель автомобиля 
    if isGetOfferClick and isGetBmwClick and not(isModelNotEmpty):
        global model 
        model = message_text        
        isModelNotEmpty = True
        reply_text = f'Введите год выпуска автомобиля'
        update.message.reply_text(text=reply_text)



    
# приветствие
def greeting_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    if user:
        name = user.first_name
    else:
        name = 'аноним'
    
    text = "Я телеграм-бот, который поможет тебе оформить страховку на автомобиль!\nВоспользуйся меню чтобы узнать подробности или перейти к оформлению!"
    reply_text = f"Привет, {name}!\n\n{text}"

    # Ответить пользователю
    return update.message.reply_text(text=reply_text, reply_markup=get_keyboard(),)

    # Записать сообщение в БД
    if text:
        add_message(user_id=user.id,
            text=text,)


# menu's handler
def callback_handler(update: Update, context: CallbackContext):
    user = update.effective_user
    callback_data = update.callback_query.data
    
    global isGetOfferClick 
    global isGetBmwClick

    photoLight=''
    photoOptimum=''
    keyboard= None
    text =''
    if callback_data == GET_OFFER:
        text = f'Для начала выберите марку автомобиля:'
        keyboard=get_maker_keyboard()
        isGetOfferClick = True
    elif callback_data == GET_PRODUCT_INFO:
        text = f'Условия продукта страхования. Для вызова главного меню напишите любое сообщение.'
        photoLight = 'https://github.com/JRSY23/PY/blob/master/%D0%9B%D0%B0%D0%B9%D1%82.png?raw=true'
        photoOptimum = 'https://github.com/JRSY23/PY/blob/master/%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D1%83%D0%BC.png?raw=true'
        restart_process()
    elif callback_data == GET_MANUAL:
        text = f'Инструкция \"Как пользоваться нашим ботом\" (Сделать ПДФ). Для вызова главного меню напишите любое сообщение.'    
        restart_process()
    elif callback_data == GET_BMW_MODELS:
        text = f'Введите модель автомобиля на английском (Например X5, 320i, m5, z4).' 
        global maker 
        maker = 'bmw' # Марка - BMW
        isGetBmwClick = True
    elif callback_data == CONFIRM:
        link = predict(mileage, engine_power, manufacture_year, maker, model)
        text = f'Спасибо, что выбрали наш сервис. Перейдите по ссылке для оформления полиса. {link}' 
        restart_process()
    elif callback_data == CANCEL:
        text = f'Очень жаль:(' 
        restart_process()
    else:
        text = 'Произошла ошибка'
        restart_process()

    update.effective_message.reply_text(text=text,reply_markup=keyboard)
    if (photoLight!='' and photoOptimum !=''):
         update.effective_message.reply_photo(photo=photoLight,)
         update.effective_message.reply_photo(photo=photoOptimum,)

# restart process and clean data
def restart_process():
    global isFirstMessageSend
    global isGetOfferClick 
    global isGetBmwClick
    global isModelNotEmpty
    global isMileageNotEmpty 
    global isManufactureYearNotEmpty
    global isEnginePowerNotEmpty
    isFirstMessageSend = False
    isGetOfferClick = False
    isGetBmwClick = False
    isModelNotEmpty = False
    isMileageNotEmpty = False
    isManufactureYearNotEmpty = False
    isEnginePowerNotEmpty = False


def main():
    logging.basicConfig(filename="kaskoBot.log", level=logging.INFO)
    print('Start KaskoBot')
    logger.log(level=logging.INFO, msg='Start KaskoBot')

    # объявление бота и тд
    req = Request(connect_timeout=0.5,
        read_timeout=1.0,)
    bot = Bot(token='1201845137:AAHh2QcXRHAKcJ1HK7LVUroMlaRLSSYQj9s',
        request=req,
        base_url='https://telegg.ru/orig/bot',)
    updater = Updater(bot=bot,
        use_context=True,)

    # Проверить что бот корректно подключился к Telegram API
    info = bot.get_me()
    print(f'Bot info: {info}')
    logger.log(level=logging.INFO, msg=f'Bot info: {info}')

    # Подключиться к СУБД
    #init_db()

    # Навесить обработчики команд
    updater.dispatcher.add_handler(MessageHandler(Filters.all, message_handler))
    updater.dispatcher.add_handler(CallbackQueryHandler(callback_handler))

    # Начать бесконечную обработку входящих сообщений
    updater.start_polling()
    updater.idle()
    print('Stop KaskoBot')
    logger.log(level=logging.INFO, msg='Stop KaskoBot')


if __name__ == '__main__':
    main()