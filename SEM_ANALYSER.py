import streamlit as st
import cv2
import numpy as np
import os
import io 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import datetime

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(layout="wide", page_title="SEM Analyzer")

# --- 2. ИНИЦИАЛИЗАЦИЯ SESSION STATE (Вместо глобальных переменных) ---
# В Streamlit код перезапускается при каждом действии. 
# Чтобы данные не терялись, храним их в st.session_state.
if 'data_pool' not in st.session_state:
    st.session_state['data_pool'] = [] # Аналог твоего session_data = []

# ================= 3. BACKEND: LOGIC (Твой класс) =================
# Класс остается без изменений, логика чистая
class SEMImageProcessor:
    def __init__(self, image_path, scale_bar_nm, scale_bar_pixels, crop_bottom=0):
        self.image_path = image_path
        self.nm_per_pixel = scale_bar_nm / scale_bar_pixels if scale_bar_pixels > 0 else 0
        self.crop_bottom = crop_bottom
        self.original_img = None

    def load_and_prepare(self):
        try:
            self.image_path.seek(0) # <--- ПЕРЕНЕСТИ СЮДА (ОБЯЗАТЕЛЬНО ДО read)
            file_bytes = np.asarray(bytearray(self.image_path.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img is None: return None
            if self.crop_bottom > 0: 
                img = img[:-self.crop_bottom, :]
            self.original_img = img
            return img
        except Exception as e:
            raise IOError(f"Ошибка загрузки изображения: {e}")
            return None

    def segment(self, method_id=2, params=None):
        if self.original_img is None: return None
        if params is None: params = {}
        
        # Определяем тип порога (обычный или инвертированный)
        invert = params.get('invert', False)
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        blur_size = params.get('blur_size', 3)
        blur = cv2.medianBlur(self.original_img, blur_size if blur_size % 2 != 0 else blur_size + 1)
        
        mask = None
        if method_id == 1: # Otsu
            _, mask = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
        elif method_id == 2: # Adaptive
            blk = params.get('adaptive_block_size', 11)
            if blk % 2 == 0: blk += 1
            mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        thresh_type, blk, params.get('adaptive_c', 2))
        elif method_id == 3: # Top-Hat
            k_size = params.get('tophat_kernel', 15)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k)
            _, mask = cv2.threshold(tophat, 0, 255, thresh_type + cv2.THRESH_OTSU)
        
        # Морфология с безопасным получением параметров
        if params.get('enable_opening'):
            s = params.get('open_kernel', 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if params.get('enable_closing'):
            s = params.get('close_kernel', 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def analyze_particles(self, mask, min_nm, max_nm):
        if mask is None: return {}, [], []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_data, acc_cnts, rej_cnts = [], [], []
        
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px <= 1: continue
            d_nm = 2 * np.sqrt(area_px / np.pi) * self.nm_per_pixel
            
            if min_nm <= d_nm <= max_nm:
                valid_data.append(d_nm)
                acc_cnts.append(cnt)
            else: 
                rej_cnts.append(cnt)
                
        return {'diameters': valid_data, 'count': len(valid_data)}, acc_cnts, rej_cnts

# ================= 4. CONFIG & SETUP =================

# ================= 5. ГЕОМЕТРИЯ И ИНТЕРФЕЙС (STREAMLIT) =================

# Функция для расчетов остается такой же (Backend)
def get_weighted_stats(data, weights):
    """Вычисляет взвешенное среднее и медиану"""
    if len(data) == 0: return 0.0, 0.0
    mean = np.average(data, weights=weights)
    
    sort_idx = np.argsort(data)
    d_sorted = data[sort_idx]
    w_sorted = weights[sort_idx]
    cumsum_w = np.cumsum(w_sorted)
    median = d_sorted[np.searchsorted(cumsum_w, cumsum_w[-1] / 2.0)]
    return mean, median

# --- БОКОВАЯ ПАНЕЛЬ (SIDEBAR) ---

st.sidebar.header("📂 Загрузка данных")

# 1. Загрузчик файлов (теперь это наш источник данных)
uploaded_files = st.sidebar.file_uploader(
    "Загрузите СЭМ изображения", 
    type=['png', 'jpg', 'jpeg', 'tif'], 
    accept_multiple_files=True
)

# 2. Проверка: если файлов нет, останавливаем выполнение и просим загрузить
if not uploaded_files:
    st.info("👈 Пожалуйста, загрузите СЭМ-снимки через панель слева.")
    st.stop()

# 3. Выбор конкретного файла из списка ЗАГРУЖЕННЫХ
# Мы используем format_func, чтобы в списке красиво отображались имена файлов
selected_file = st.sidebar.selectbox(
    "Выберите файл для настройки", 
    options=uploaded_files, 
    format_func=lambda x: x.name
)

# Для удобства создаем алиас (чтобы старый код не ломался)
image_full_path = selected_file
st.sidebar.divider()

st.sidebar.header("📏 Калибровка (Scale Bar)")
col_nm, col_px, col_crop = st.sidebar.columns(3)

with col_nm:
    # Важно: имя переменной должно быть scale_nm
    scale_nm = st.number_input("Размер (нм)", min_value=1.0, value=2000.0, step=100.0)

with col_px:
    # Важно: имя переменной должно быть scale_px
    scale_px = st.number_input("В пикселях", min_value=1, value=122)

with col_crop: 
    crop_val = st.number_input("Обрезка низа (px)", 0, 500, 88)

st.sidebar.divider()

# 3. Метод сегментации

st.sidebar.header("🔬 Настройки обработки")
method_name = st.sidebar.selectbox("Метод порога", ["Adaptive", "Top-Hat", "Otsu"])
method_id = {"Otsu": 1, "Adaptive": 2, "Top-Hat": 3}[method_name]

# 1. ОБЯЗАТЕЛЬНО: Создаем пустой словарь перед условиями!
# --- В блоке настроек обработки ---
params = {
    'adaptive_block_size': 203,
    'adaptive_c': 0,
    'tophat_kernel': 113,
    'blur_size': 3,
    'invert': False
}

params['invert'] = st.sidebar.checkbox("Инвертировать маску (ч/б)", value=False)
params['blur_size'] = st.sidebar.slider("Размытие (Median Blur)", 1, 15, 3, step=2)

if method_name == "Adaptive":
    params['adaptive_block_size'] = st.sidebar.slider("Block Size", 3, 1001, 203, step=2)
    params['adaptive_c'] = st.sidebar.slider("C (Constant)", -30, 30, 0)
elif method_name == "Top-Hat":
    params['tophat_kernel'] = st.sidebar.slider("Top-Hat Kernel", 3, 501, 113, step=2)

st.sidebar.divider()

with st.sidebar.expander("Морфология (Шум/Склейка)"):
    params['enable_opening'] = st.checkbox("Удалить мелкий шум (Opening)", value=False)
    params['open_kernel'] = st.slider("Open Kernel size", 3, 51, 3, step=2) # старт с 3
    
    params['enable_closing'] = st.checkbox("Заполнить пустоты (Closing)", value=True)
    params['close_kernel'] = st.slider("Close Kernel size", 3, 51, 3, step=2) # старт с 3

# 4. Фильтры размера
# Замени блок фильтров на этот:
st.sidebar.header("📏 Фильтры частиц")
min_val, max_val = st.sidebar.slider(
    "Диапазон размеров (нм)", 
    0, 10000, (10, 2000) # Возвращает кортеж (min, max)
)
min_size, max_size = min_val, max_val

# 5. Режим статистики
weight_mode = st.sidebar.radio("Режим нормировки", ["Count", "Area", "Volume"], horizontal=True)
show_w_mean = st.sidebar.checkbox("Взвешенное среднее", value=True)
show_w_median = st.sidebar.checkbox("Взвешенная медиана", value=True)
show_u_mean = st.sidebar.checkbox("Арифм. среднее (простое)", value=False)
show_kde_line = st.sidebar.checkbox("Линия KDE (тренд)", value=True)

# 6. Кнопки управления (внизу сайдбара)

st.sidebar.divider()

st.sidebar.info(f"В пуле сейчас: {len(st.session_state['data_pool'])} фото")

btn_add = st.sidebar.button("➕ ДОБАВИТЬ В ПУЛ", use_container_width=True)

# Сброс пула (используем session_state)
if st.sidebar.button("🗑 СБРОСИТЬ ПУЛ", use_container_width=True):
    st.session_state['data_pool'] = []
    st.toast("Пул данных очищен!")

# ================= 5. ГЛАВНАЯ ЛОГИКА ОБРАБОТКИ (MAIN AREA) =================

st.title("🔬 SEM Microstructure Analyzer")

# Выполняем обработку текущего изображения
# (Параметры берутся напрямую из виджетов sidebar, определенных в прошлой части)
proc = SEMImageProcessor(image_full_path, scale_nm, scale_px, crop_bottom=crop_val)
img = proc.load_and_prepare()

if img is not None:
    mask = proc.segment(method_id=method_id, params=params)
    stats, acc_cnts, rej_cnts = proc.analyze_particles(mask, min_size, max_size)

    st.subheader(f"Анализ: {selected_file.name}")
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        # Создаем фигуру БЕЗ использования plt.close() внутри этой функции
        fig1, ax1 = plt.subplots()
        ax1.imshow(mask, cmap='gray')
        ax1.set_title("Маска (Binary)")
        ax1.axis('off')
        st.pyplot(fig1, clear_figure=True) # Параметр clear_figure=True очистит память сам

    with col_img2:
        fig2, ax2 = plt.subplots()
        res_img = cv2.cvtColor(proc.original_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(res_img, acc_cnts, -1, (0, 255, 0), 2)
        cv2.drawContours(res_img, rej_cnts, -1, (255, 0, 0), 1)
        ax2.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Найдено: {stats['count']}")
        ax2.axis('off')
        st.pyplot(fig2, clear_figure=True)

    # --- ОБРАБОТКА КНОПКИ "ДОБАВИТЬ В ПУЛ" ---
    if btn_add:
        h, w = proc.original_img.shape
        nm_px = scale_nm / scale_px
        new_entry = {
            'diameters': stats['diameters'],
            'area': (w * nm_px) * (h * nm_px),
            'filename': selected_file
        }
        st.session_state['data_pool'].append(new_entry)
        st.success(f"Добавлено {stats['count']} частиц из {selected_file.name}")

# ================= 6. ОБЩАЯ СТАТИСТИКА (POOL ANALYSIS) =================

if st.session_state['data_pool']:
    st.divider()
    st.subheader(f"📊 Общая статистика ({len(st.session_state['data_pool'])} изобр.)")
    
    all_d = []
    all_w = []
    
    # Сбор данных из всех накопленных записей
    for entry in st.session_state['data_pool']:
        d = np.array(entry['diameters'])
        if weight_mode == 'Count':
            w = np.ones_like(d) * (1e6 / entry['area'])
        elif weight_mode == 'Area':
            w = (np.pi * (d/2)**2) / entry['area']
        else: # Volume
            w = ((4/3) * np.pi * (d/2)**3) / entry['area']
        
        all_d.extend(d)
        all_w.extend(w)
        
    all_d = np.array(all_d)
    all_w = np.array(all_w)
    
    # Расчет Mean/Median
    w_mean, w_median = get_weighted_stats(all_d, all_w)
    
# 1. Настройка стиля (локально для этого графика)
    with plt.style.context("seaborn-v0_8-muted"): # Или любой другой стиль
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        
        # Настройка шрифтов для публикации
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

        # 2. Правильные Логарифмические бины
        # Добавляем небольшой отступ (0.9 и 1.1), чтобы крайние значения не прилипали к границам
        bin_min = max(all_d.min() * 0.9, 0.1) 
        bin_max = all_d.max() * 1.1
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 50)
        
        # 3. Гистограмма
        ax_hist.hist(all_d, bins=bins, weights=all_w, color='teal', alpha=0.5, 
                     edgecolor='black', density=True, label='Histogram')

        # Простая статистика (без учета весов) для сравнения, если выбрано
        if show_u_mean:
            u_mean = np.mean(all_d)
            ax_hist.axvline(u_mean, color='gray', ls='--', alpha=0.6, label=f'Simple Mean: {u_mean:.1f}')
    
        # Взвешенная статистика
        if show_w_mean:
            ax_hist.axvline(w_mean, color='red', ls='-', lw=2, label=f'W. Mean: {w_mean:.1f} nm')
        
        if show_w_median:
            ax_hist.axvline(w_median, color='blue', ls=':', lw=2, label=f'W. Median: {w_median:.1f} nm')
        
        # KDE (тренд)
        if show_kde_line and len(all_d) > 3:
            kde = gaussian_kde(np.log10(all_d), weights=all_w)
            x_grid = np.logspace(np.log10(bin_min), np.log10(bin_max), 300)
            y_kde = kde(np.log10(x_grid)) / (x_grid * np.log(10))
            ax_hist.plot(x_grid, y_kde, color='black', lw=2, label='KDE Trend')
    
        # Не забудь обновить легенду, чтобы она видела только включенные линии
        if any([show_w_mean, show_w_median, show_u_mean, show_kde_line]):
            ax_hist.legend(loc='upper right', frameon=True)

        # 6. Оформление осей
        ax_hist.set_xscale('log')
        ax_hist.set_xlabel('Diameter (nm)', fontweight='bold', fontsize=12)
        ax_hist.set_ylabel(f'Density ({weight_mode}-weighted)', fontweight='bold', fontsize=12)
        ax_hist.grid(True, which='both', alpha=0.2, ls='-')
        
        # 7. ЛЕГЕНДА (Вызываем ПЕРЕД сохранением)
        ax_hist.legend(loc='upper right', frameon=True, fontsize=10)
        
        # 8. ОТОБРАЖЕНИЕ В STREAMLIT
        st.pyplot(fig_hist)

        # 9. СОХРАНЕНИЕ В PDF (Теперь всё попадет в файл)
        buf = io.BytesIO()
        # bbox_inches='tight' важен, чтобы легенда не обрезалась
        fig_hist.savefig(buf, format="pdf", dpi=600, bbox_inches='tight')
        
        st.download_button(
            label="💾 СКАЧАТЬ ПОЛНЫЙ ОТЧЕТ (PDF)",
            data=buf.getvalue(),
            file_name=f"SEM_Report_{datetime.datetime.now().strftime('%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        plt.close(fig_hist)
    
    # Кнопка скачивания CSV данных
    df_pool = pd.DataFrame(all_d, columns=['Diameter_nm'])
    csv = df_pool.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать данные (CSV)", data=csv, file_name="sem_data.csv", mime='text/csv')

else:
    st.info("Пул данных пуст. Настройте параметры и нажмите 'Добавить в пул' в боковой панели.")


    