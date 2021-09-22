def load_kaggle_data(kaggle_config_dir, path_from, path_to):
    
    """ Устанавливает API KAGGLE, 
        определяет переменную окружения KAGGLE_CONFIG_DIR для хранения пути к каталогу конфигурации KAGGLE,
        переопределяет права на доступ к информации в каталоге конфигурации KAGGLE,
        загружает датасет и распаковывает его.
        Выводит версию API."""
    
    import os
    from subprocess import run, STDOUT, PIPE
    
    # установка kaggle API
    output = run('pip freeze'.split(), stdout=PIPE, stderr=STDOUT, text=True)
    if output.stdout.find('kaggle') == -1:
        output = run('pip install kaggle'.split(), stdout = PIPE, stderr = STDOUT, text = True)
    
    # установка переменной окружения KAGGLE_CONFIG_DIR (конфигурация для доступа к ресурсам KAGGLE) 
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir
    print(f'KAGGLE_CONFIG_DIR = {os.environ["KAGGLE_CONFIG_DIR"]}')
    
    # переопределение прав на доступ к файлу kaggle.json (ключ)
    path_to_key = kaggle_config_dir + '/kaggle.json'
    os.system('chmod 600 ' + path_to_key)

    # вывод версии API KAGGLE
    output = run('kaggle --version'.split(), stdout=PIPE, stderr=STDOUT, text=True)
    print(output.stdout)

    # загрузка данных по винам
    os.system('kaggle datasets download ' + path_from + ' -p ' + path_to + ' --unzip')
    
    return

if __name__ == "__main__":
    pass

