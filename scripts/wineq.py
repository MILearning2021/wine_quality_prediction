#!/usr/bin/env python
# coding: utf-8

# <a id="Table-Of-Contents"></a>
# # Содержание
# * [1. Описание задачи](#Task-Details)
# * [2. Анализ данных](#Data-Analysis)
#     * [2.1 Импорт библиотек](#Importing-Libraries)
#     * [2.2 Вспомогательные функции](#Helper-functions)
#     * [2.3 Описание переменных](#Feature-Description)
#     * [2.4 Чтение данных и превый взгляд на них](#Read-in-Data)
# * [3. Решения по переменным](#Solutions-for-variables)
# * [4. Вопросы заказчика](#Customer-Questions)
#     * [4.1 Алгоритм выбора актуальных задаче предсказания качества вина признаков](#Outl-detection)
#     * [4.2 Алгоритм определения выбросов в данных для выделения классов вин "excellent" и "poor"](#Feature-selection)
#     * [4.3 Алгоритм предсказания качества вина](#ML)
# * [5. Выводы](#Conclusion)

# <a id="Task-Details"></a>
# # 1. Описание задачи

# **The goal** - to predict the quality of wines depending on their physicochemical composition.
# 
# 
# The dataset was downloaded from the UCI Machine Learning Repository.
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. The reference [Cortez et al., 2009]. 
# Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no 
# data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch 
# more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. 
# Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# Two datasets were combined and few values were randomly removed.
# 
# * [Содержание](#Table-Of-Contents)

# <a id="Data-Analysis"></a>
# # 2. Анализ данных

# <a id="Importing-Libraries"></a>
# # 2.1 Импорт библиотек

# In[149]:


import os
import sys
import pickle
import bz2

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.cluster import DBSCAN

from imblearn.over_sampling import SMOTE


# In[150]:


pd.set_option('display.max_rows', None)


# * [Содержание](#Table-Of-Contents)

# <a id="Helper-functions"></a>
# # 2.2 Вспомогательные функции

# In[151]:


# загрузка данных с KAGGLE
path_to_local_data = os.path.abspath(os.path.relpath(os.getcwd(), start = 'wine_quality_prediction'))
path_to_kaggle_data = 'rajyellow46/wine-quality'
path_to_KAGGLE_CONFIG_DIR = '/Users/ivanpetrov/Desktop/DataScientiest/KAGGLE_CONFIG_DIR'

sys.path.append(path_to_local_data + '/scripts')

from load_kaggle_data import load_kaggle_data

load_kaggle_data(path_to_KAGGLE_CONFIG_DIR, path_to_kaggle_data, path_to_local_data + '/data')


# In[171]:


# вывод нескольких однотипных графиков
def plot_graph(df_, list_x, list_y = None, titles_ = None, figsize_ = (18, 1.5), type_ = 'boxplot', xbound_ = None):
    axes = []
    fig, axs = plt.subplots(1, len(list_x), sharey = False, figsize = figsize_)      
    if len(list_x) > 1:
        axes = axs
    else:
        axes.append(axs)
    for ind, ax in enumerate(axes): 
        if type_ == 'boxplot':
            ax.boxplot(df_[list_x[ind]], False, 'r*', vert = False)
            ax.set_title(list_x[ind])
            ax.set_xbound(lower = xbound_[ind][0], upper = xbound_[ind][1])
            ax.set_yticklabels([])
        elif type_ == 'scatter':
            ax.scatter(df_[list_x[ind]], df_[list_y[ind]], marker = '.')
            ax.set_ylabel(list_y[ind])
            ax.set_xlabel(list_x[ind])
        elif type_ == 'hist':
            ax.hist(df_[list_x[ind]], bins = 30)
            #ax.set_xlabel(list_x[ind])
            ax.set_xbound(lower = xbound_[ind][0], upper = xbound_[ind][1])
            ax.set_yticklabels([])
    plt.savefig(path_to_local_data + '/results/' + type_ + '_' + df_.name)
    plt.show()
    return
    
# вывод "тепловой" карты
def heatmap(df_, size, df_name):
    plt.figure(figsize = size)
    sns.heatmap(df_, cmap = 'Blues', annot = True, cbar = True)
    plt.savefig(path_to_local_data + '/results/heatmap_' + df_name)
    plt.show()
    return
    
# выбросы
def outliers(df_, column_, contamination_ = .5):
    pred = IsolationForest(n_estimators = 100, contamination = contamination_, random_state = 42).fit_predict(df_[column_])
    print(f'{column_}: {len(np.where(pred == -1)[0])} сэмпл., {np.where(pred == -1)[0]}')
    if len(column_) == 1:
        plt.figure(figsize = (12, 3))
        plt.scatter(df_[column_], df_['quality'], c = 'b')
        plt.scatter(df_.iloc[np.where(pred == -1)[0]][column_], df_.iloc[np.where(pred == -1)[0]]['quality'], c = 'r')
        plt.savefig(path_to_local_data + '/results/outliers' + '_' + df_.name + '_' + column_[0].replace(' ', '_'))
        plt.show()      
    return df_.iloc[np.where(pred == -1)[0]]

# формирование тестового и тренировочного наборов данных
def split_and_smote(df_, oversampling_ = True):
    X_tr, X_t, y_tr, y_t = train_test_split(df_.drop(columns = ['quality']), df_['quality'], test_size = 0.2, shuffle = True, \
                                            stratify = df_['quality'], random_state = 42)    
    if oversampling_:
        print(f'Размерность до применения SMOTE: X_tr {X_tr.shape}, y_tr {y_tr.shape}')
        smote = SMOTE(sampling_strategy = 'not majority', k_neighbors = 2, random_state = 42)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        print(f'Размерность после применения SMOTE: X_tr {X_tr.shape}, y_tr {y_tr.shape}')
    return X_tr, X_t, y_tr, y_t

# тестирование модели
def evaluate_solution(y_t, pred_proba, pred_labels, labels = None):
    print(f'ROC AUC на тестовом наборе: {roc_auc_score(y_t, pred_proba, multi_class = "ovr")}')
    print('---------------------------------------------------------')
    print(f'Отчет о классификации:\n\n{classification_report(y_t, pred_labels, labels = labels)}')
    print('---------------------------------------------------------')
    print(f'Матрица ошибок:\n\n{confusion_matrix(y_t, pred_labels)}')
    print('---------------------------------------------------------')
    return

# подбор гиперпараметров модели
def GridSearchCrossVal(estimator_, params_, X_tr, y_tr):
    gscv = GridSearchCV(estimator_, params_)
    gscv.fit(X_tr, y_tr)
    return gscv

# выбор лучшей модели
def models_results(params_, X_tr, X_t, y_tr, y_t, df_name):
    results = {}
    for key in params_:
        print(f'----------------------\n{key}\n----------------------')
        # подбор гиперпараметров модели
        gscv = GridSearchCrossVal(params_.get(key)[0], params_.get(key)[1], X_tr, y_tr)
        # вывод результатов тестирования модели
        evaluate_solution(y_t, gscv.best_estimator_.predict_proba(X_t), gscv.best_estimator_.predict(X_t))
        # сохраним обученную модель в словарик на всякий случай
        results[key] = gscv
        # сохраним обученную модель в файлик на всякий случай
        with open(path_to_local_data + '/models/' + df_name + '_' + key, 'wb') as file:
            pickle.dump(bz2.compress(pickle.dumps(gscv)), file)            
    return results


# * [Содержание](#Table-Of-Contents)

# <a id="Feature-Description"></a>
# # 2.3 Описание переменных

# **Input** variables (based on physicochemical tests):  
#     1 - fixed acidity: The predominant fixed acids found in wines are tartaric, malic, citric, and succinic. Their respective levels 
# found in wine can vary greatly and wines with high acidity might taste sour which effect the quality.  
#     2 - volatile acidity: The amount of acetic acid in wine, which at too high of levels can lead to an unpleasant taste.  
#     3 - citric acid: Similar as above, very less in wine.  
#     4 - residual sugar: This effect the sweetness of the wine.  
#     5 - chlorides: This will stay in the form of potassium salt in wine.  
#     6 - free sulfur dioxide: The SO2 exists in equilibrium, too much will effect the health.  
#     7 - total sulfur dioxide: Amount of free and bound forms of S02  
#     8 - density: During the fermentation process, fructose is converted into alcohol, the density is reduced also influence the taste.  
#     9 - pH: Describes how acidic or basic a wine is, most wines are between 3–4. Monitoring the total acidity of red wine is an important 
# indicator of its quality and an important influencing factor of its taste.  
#     10 - sulphates: It has the effects of selection, clarification, anti-oxidation, acidification, dissolution, etc. it used to keep it 
# fresh and taste, and maintain the wine flavor.  
#     11 - alcohol: The percent alcohol content of the wine  
# **Target** variable (based on sensory data):   
#     12 - quality (score between 0 and 10)
#   
#   * [Содержание](#Table-Of-Contents)

# <a id="Read-in-Data"></a>
# # 2.4 Чтение данных и первый взгляд на них

# In[153]:


# чтение данных по винам
df = pd.read_csv(path_to_local_data + '/data/winequalityN.csv', sep = ',')


# In[154]:


df.info()


# In[155]:


np.unique(df['type'], return_counts = True)


# Переменная типа вина "type" не сбалансирована (наблюдений красного вина в 3 раза меньше).  
# Учитывая вероятное наличие систематических отклонений в сэмплах в заисимости от типа вина, следует проводить 
# анализ данных в разрезе типа вина раздельно.

#   * [Содержание](#Table-Of-Contents)

# <a id="Solutions-for-variables"></a>
# # 3. Решения по переменным

# In[156]:


# NaN
print(f'Сэмплов с "NaN": {len(df[df.isna().any(1)])} или {round(len(df[df.isna().any(1)]) * 100.0 / len(df), 2)}%')


# Количество сэмплов с 'NaN' около 0,5%. Решаем удалить их.

# In[157]:


df.dropna(axis = 0, inplace = True)
print(f'Сэмплов с "NaN": {len(df[df.isna().any(1)])} или {round(len(df[df.isna().any(1)]) * 100.0 / len(df), 2)}%')


# In[158]:


# разделим данные по типу вина red/white
df_red = df[df['type']=='red'].drop(columns = ['type'])
df_red.name = 'red'
df_white = df[df['type']=='white'].drop(columns = ['type'])
df_white.name = 'white'


# In[159]:


df_red.describe()


# In[160]:


df_white.describe()


# Анализ статистик признаков в разрезе типа вина показывает наличие существенных различий большинства характеристик вин в зависимости 
# от типа "белое/красное".

# In[161]:


# целевая переменная
print(f'Красные вина: {np.unique(df_red["quality"], return_counts = True)}')
print(f'Белые вина: {np.unique(df_white["quality"], return_counts = True)}')


# Целевая переменная в обоих наборах данных не сбалансирована: налицо явное преобладание вин со средней оценкой.

# * [Содержание](#Table-Of-Contents)

# <a id="Customer-Questions"></a>
# # 4. Вопросы заказчика

# <a id="Feature-selection"></a>
# # 4.1 Алгоритм выбора признаков, актуальных задаче предсказания качества вина

# Проблему выбора актуальных признаков попробуем решить с помощью методики многофакторного дисперсионного анализа, который позволит нам 
# проверить гипотезы о важности тех или иных признаков для наших наборов данных по винам.

# Первым шагом попробуем определить наличие проблемы мультиколинеарности в наборах данных по винам.

# Как видно из информации о каждом столбце выше, значение рН зависит от количества кислоты (acidity) в вине, а плотность (density) от 
# количества алкоголя (alcohol). Свободный диоксид серы (free sulfur dioxide) входит в состав общего диоксида серы (total sulfur dioxide). 
# Citric acid является составной частью fixed acidity.

# Проверим наличие взаимосвязи перечисленных выше признаков.

# In[162]:


# красное вино
list_x = ['free sulfur dioxide', 'alcohol', 'fixed acidity', 'citric acid']
list_y = ['total sulfur dioxide', 'density', 'pH', 'fixed acidity']
grid_columns = 2
for x in range(0, len(list_x) // grid_columns):
    plot_graph(df_red, list_x[x * grid_columns : (x * grid_columns) + grid_columns], \
               list_y[x * grid_columns : (x * grid_columns) + grid_columns], titles_ = None, figsize_ = (18, 4), type_ = 'scatter')
pass


# In[165]:


heatmap(df_red[np.unique(list_x + list_y)].corr(), (7, 4), df_red.name)


# Значимая корреляция признаков для красного вина присутствует. 

# In[166]:


# белое вино
list_x = ['free sulfur dioxide', 'alcohol', 'fixed acidity', 'citric acid']
list_y = ['total sulfur dioxide', 'density', 'pH', 'fixed acidity']
grid_columns = 2
for x in range(0, len(list_x) // grid_columns):
    plot_graph(df_white, list_x[x * grid_columns : (x * grid_columns) + grid_columns], \
               list_y[x * grid_columns : (x * grid_columns) + grid_columns], titles_ = None, figsize_ = (18, 4), type_ = 'scatter')
pass


# In[167]:


heatmap(df_white[np.unique(list_x + list_y)].corr(), (7, 4), df_white.name)


# Картинка несколько изменилась, но значимая корреляция признаков для белого вина также присутствует.

# Для исключения проблемы мультиколинеарности уберем несколько признаков из наборов данных, а именно: 'citric acid', 'density', 
# 'free sulfur dioxode', 'pH'. Оставшиеся признаки будут участвовать в построении линейной модели и определении их значимости для 
# предсказания целевой переменной 'quality'.

# Примем за нулевую гипотезу отсутствие значимости оставшихся признаков для определения 'quality'.  
# Уровень значимости равен 0.05.

# In[168]:


# красное вино
from statsmodels.formula.api import ols

data = df_red[['fixed acidity', 'volatile acidity', 'residual sugar', 
               'chlorides', 'total sulfur dioxide','sulphates', 'alcohol', 
               'quality']].rename(columns = {'fixed acidity' : 'fixed_acidity', \
                                             'volatile acidity' : 'volatile_acidity', \
                                             'residual sugar' : 'residual_sugar', \
                                             'total sulfur dioxide' : 'total_sulfur_dioxide' \
                                        })

print(ols('quality ~ fixed_acidity + volatile_acidity + residual_sugar + chlorides + total_sulfur_dioxide' + ' + sulphates + alcohol', \
          data = data).fit().summary().tables[1])


# Все признаки модели, кроме 'residual sugar', имеют уровень значимости (P>|t|) меньше 0.05, что позволяет отклонить нулевую гипотезу и сделать вывод о наличии значимости указанных признаков для определения 'quality' красного вина.  
# **Таким образом, значимыми признаками являются:  
# 'fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide','sulphates', 'alcohol'.**

# In[169]:


# белое вино
data = df_white[['fixed acidity', 'volatile acidity', 'residual sugar', 
                 'chlorides', 'total sulfur dioxide','sulphates', 'alcohol', 
                 'quality']].rename(columns = {'fixed acidity' : 'fixed_acidity', \
                                               'volatile acidity' : 'volatile_acidity', \
                                               'residual sugar' : 'residual_sugar', \
                                               'total sulfur dioxide' : 'total_sulfur_dioxide' \
                                        })

print(ols('quality ~ fixed_acidity + volatile_acidity + residual_sugar + chlorides + total_sulfur_dioxide' + ' + sulphates + alcohol', \
          data = data).fit().summary().tables[1])


# Все признаки модели, кроме 'total sulfur dioxide', имеют уровень значимости (P>|t|) меньше 0.05, что позволяет отклонить нулевую гипотезу 
# и сделать вывод о наличии значимости указанных признаков для определения 'quality' белого вина.  
# Уровень значимости признака 'chlorides' очень близок к пороговому значению в 0.05. Учитывая данные, полученные при исследовании предметной 
# области, можно утверждать, что и признак 'chlorides' также имеет значимое влияние на качество вина.  
# **Таким образом, значимыми признаками являются:  
# 'fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar','sulphates', 'alcohol'.**

# * [Содержание](#Table-Of-Contents)

# <a id="Outl-detection"></a>
# # 4.2 Алгоритм определения выбросов в данных для выделения классов вин "excellent" и "poor"

# Для определения выбросов воспользуемся алгоритмом IsolationForest. Представим данные в разрезе классов 'quality' экспертной оценки. 
# Для обоих типов вин исследуем по два самых важных параметра: 'volatile acidity' и 'chlorides'.  

# In[172]:


# красное вино
_ = outliers(df_red, ['chlorides'], .0135)


# In[173]:


_ = outliers(df_red, ['volatile acidity'], .0035)


# In[174]:


# белое вино
_ = outliers(df_white, ['chlorides'], .0015)


# In[175]:


_ = outliers(df_white, ['volatile acidity'], .0012)


# Анализ отдельных признаков дает возможность определить выбросы значений признаков (для наших датасетов малые значения выбросов не 
# определяются в связи с наличием смещения значений параметров в сторону 0.0).  
# Однако, делать вывод о качестве вина по отдельным признакам не логично, учитывая комплексный характер характеристик вин.

# Теперь, используя тот же алгоритм определения выбросов IsolationForest, попробуем определить выбросы в пространстве нескольких признаков, 
# а именно всех тех, важность которых определена в п.4.1:  
#  - для белого вина 'fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar','sulphates', 'alcohol'  
#  - для красного вина 'fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide','sulphates', 'alcohol'

# In[176]:


# красное вино
outliers(df_red, ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide','sulphates', \
        'alcohol'], .02).sort_values(by = 'quality')


# In[177]:


# белое вино
outliers(df_white, ['fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar','sulphates', \
        'alcohol'], .005).sort_values(by = 'quality')


# Анализ выбросов по нескольким значимым признака одновременно, в целом, позволяет сделать несколько выводов:  
# - экспертная оценка качества отражает либо преобладание значения одного параметра, либо некоторое сочетание значений нескольких параметров  
# - экспертная оценка качества сильно снижается при увеличении значений 'volatile acidity' и 'chlorides', либо одного их этих параметров
# - качественная оценка вина 'строгими' методами вряд ли возможна

# * [Содержание](#Table-Of-Contents)

# <a id="ML"></a>
# # 4.3 Алгоритм предсказания качества вина

# Для предсказания качества вина - целевой переменной 'quality' - воспользуемся алгоритмами классификации, невосприимчивыми к дисбалансу 
# целевой переменной, а именно: RandomForestClassifier, DecisionTreeClassifier, ExtraTreeClassifier.  Состав признаков возьмем исходный.

# In[180]:


# словарь параметров наших моделей
params = dict(DecisionTreeClassifier = \
                [DecisionTreeClassifier(), \
                 dict(max_depth = np.linspace(3, 10, 8, dtype = 'int'), splitter = ['random', 'best'], \
                      max_features = ['auto', 'sqrt', 'log2'])], \
              ExtraTreeClassifier = \
                [ExtraTreeClassifier(), \
                 dict(max_depth = np.linspace(3, 10, 8, dtype = 'int'), splitter = ['random', 'best'], \
                      max_features = ['auto', 'sqrt', 'log2'])], \
              RandomForestClassifier = \
                [RandomForestClassifier(), \
                 dict(n_estimators = np.linspace(200, 1000, 20, dtype = 'int'), max_features = ['auto', 'sqrt', 'log2'])])


# In[181]:


# без применения оверсэмплинга
# красное вино
X_tr, X_t, y_tr, y_t = split_and_smote(df_red, oversampling_ = False)


# In[183]:


get_ipython().run_cell_magic('time', '', 'red_gscv_results = models_results(params, X_tr, X_t, y_tr, y_t, df_red.name)')


# In[184]:


# белое вино
X_tr, X_t, y_tr, y_t = split_and_smote(df_white, oversampling_ = False)


# In[185]:


get_ipython().run_cell_magic('time', '', 'white_gscv_results = models_results(params, X_tr, X_t, y_tr, y_t, df_white.name)')


# In[186]:


# с применением оверсэмплинга
# красное вино
X_tr, X_t, y_tr, y_t = split_and_smote(df_red, oversampling_ = True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "red_gscv_results = models_results(params, X_tr, X_t, y_tr, y_t, df_red.name + '_' + 'ovrsampl')")


# In[ ]:


# белое вино
X_tr, X_t, y_tr, y_t = split_and_smote(df_white, oversampling_ = True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "white_gscv_results = models_results(params, X_tr, X_t, y_tr, y_t, \
                             df_white.name + '_' + 'ovrsampl')")


# 1. Модель не предсказала 2 класса "3" и "9" в наборе белых вин и 2 класса "3" и "4" в наборе красных вин, т.к. было мало сэмплов из этих 
# категорий. Алгоритм балансировки целевой 
# переменной также не помог.  
# 2. По категориям "5", "6", "7", которые можно отнести к общей категории качества вин "средние", модель показала
# самый высокий, но не удовлетворительный результат: полнота и точность 0.43 - 0.82.
# 3. Наиболее проблемные зоны - границы классов.

# * [Содержание](#Table-Of-Contents)

# <a id="Conclusion"></a>
# # 5. Выводы

# 1. Актуальные для наборов данных признаки:  
# - красное вино - ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide','sulphates', 'alcohol']  
# - белое вино - ['fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar', 'sulphates', 'alcohol']  
# 2. Анализ выбросов по нескольким значимым признака одновременно, в целом, позволяет сделать несколько выводов:  
# - экспертная оценка качества отражает либо преобладание значения одного параметра, либо некоторое сочетание значений нескольких параметров  
# - экспертная оценка качества сильно снижается при увеличении значений 'volatile acidity' и 'chlorides', либо одного их этих параметров
# - качественная оценка вина 'строгими' методами вряд ли возможна
# 3. В результате работы алгоритма классификации лучшее качество предсказания целевой переменной 'quality' достигается для "средних" 
# значений 5, 6 и 7, что объясняется наличием большего количества сэмплов с этими значениями 'quality' и меньшей дисперсией значений 
# важных признаков в этих сэмплах.
# 4. Из матрицы ошибок видно, что наиболее проблемные зоны - границы классов качества вина.
# 5. Классы с малым количеством сэмплов не определились.
# 

# * [Содержание](#Table-Of-Contents)

# In[ ]:




