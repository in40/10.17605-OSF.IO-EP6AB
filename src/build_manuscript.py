"""
build_manuscript.py — bilingual (RU + EN) journal manuscript as .docx.
Journal rules: single file RU+EN; A4; margins 3.5/3.5/2.5/2.5 cm; Times New Roman
11pt; line spacing 1.5; justified; first-line indent 0.5 cm; title bold 12pt left
(<=12 words); abstract 100-120 words; keywords <=8-10; structure both langs:
Title/Authors/Abstract/Keywords/Introduction/Main body/Conclusion/References/About.
Table captions ABOVE (ГОСТ "Таблица N – Название" / "Table N. Title", no period),
per-language numbering. Figure captions BELOW. RU runs ru-RU, EN runs en-US.
Build only .docx (no PDF).
Run: .venv-docx/bin/python build_manuscript.py
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = "Times New Roman"
_LANG = "en-US"

def set_run_font(run, size=11, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rfonts.set(qn(attr), FONT)
    lang = rpr.find(qn('w:lang'))
    if lang is None:
        lang = OxmlElement('w:lang'); rpr.append(lang)
    lang.set(qn('w:val'), _LANG)
    lang.set(qn('w:eastAsia'), _LANG)

def para(doc, text, size=11, bold=False, italic=False, align='justify',
        indent=True, space_after=0, space_before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after); pf.space_before = Pt(space_before)
    if align == 'justify': pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == 'left': pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif align == 'center': pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if indent: pf.first_line_indent = Cm(0.5)
    r = p.add_run(text); set_run_font(r, size, bold, italic)
    return p

def heading(doc, text, size=12):
    return para(doc, text, size=size, bold=True, align='left', indent=False,
               space_before=10, space_after=4)

def caption_above(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text); set_run_font(r, 10, bold=True)
    return p

def add_table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].paragraphs[0].paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = hdr[i].paragraphs[0].add_run(h); set_run_font(r, 10, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].paragraphs[0].paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            r = cells[i].paragraphs[0].add_run(str(val)); set_run_font(r, 10)
    return t

def setup(doc):
    s = doc.sections[0]
    s.page_height = Cm(29.7); s.page_width = Cm(21.0)
    s.top_margin = Cm(3.5); s.bottom_margin = Cm(3.5)
    s.left_margin = Cm(2.5); s.right_margin = Cm(2.5)

def figure(doc, path, caption, width_cm=12.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(path, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = cap.add_run(caption); set_run_font(r, 9, italic=True)
    return p

# ============================================================
# RUSSIAN VERSION
# ============================================================
def russian(doc):
    global _LANG
    _LANG = "ru-RU"
    para(doc, "Активационные дельта-зонды в сравнении с TF-IDF при обнаружении prompt-инъекций: предрегистрированное исследование",
         size=12, bold=True, align='left', indent=False, space_after=6)
    para(doc, "Александр Сорокин", align='left', indent=False, space_after=2)
    para(doc, "Независимый исследователь", align='left', indent=False, space_after=2)
    para(doc, "sorokin@sorokinonline.com", align='left', indent=False, space_after=2)
    para(doc, "Финансирование не получено. Работа выполнена самостоятельно, без использования ресурсов или данных "
              "работодателя.", size=9, italic=True, align='left', indent=False, space_after=8)

    heading(doc, "Аннотация")
    para(doc, "Предрегистрированное исследование (OSF ep6ab) того, что считывается из двухпроходной активационной "
             "дельты замороженной LLM (Qwen3.5-4B) при prompt-инъекции. Метод не нов; вклад — строгая оценка без "
             "утечек данных с выводом на уровне кластеров. Получено три ключевых результата. Во-первых, согласованный по источнику и длине "
             "тест показывает, что при фиксированном контенте зонд (probe) не реагирует на саму переопределяющую "
             "обёртку (wrapper) — AUROC ≈ 0.48 на всех пяти тестовых промптах. Во-вторых, базовый метод (baseline) "
             "TF-IDF по исходному тексту превосходит дельту (0.929 против 0.895). В-третьих, кажущаяся устойчивость к "
             "избыточной защите — свойство данных, а не метода. Дельта кодирует различия контента и источника, а не "
             "намерение переопределения (override); для защитного механизма (guardrail) дешёвый текстовый метод "
             "практичнее.")

    heading(doc, "Ключевые слова")
    para(doc, "prompt-инъекция; активационные дельты; линейный зонд; избыточная защита; ложные срабатывания; "
             "предрегистрация; замороженная LLM; TF-IDF.", align='left', indent=False)

    heading(doc, "Введение")
    para(doc, "Атаки prompt-инъекций заключаются в попытке пользовательской реплики отменить или подменить системный "
             "промпт большой языковой модели. Семейство методов обнаружения на основе разности внутренних "
             "представлений (activation delta) предполагает, что намерение переопределения линейно считывается из "
             "скрытых состояний замороженной модели. Мы предзарегистрировали (OSF ep6ab, DOI 10.17605/OSF.IO/EP6AB) "
             "два вопроса. Первый — декодируемость: считывается ли намерение переопределения из дельты (гипотезы "
             "H1–H3). Второй — ценность: добавляет ли внутреннее состояние что-либо сверх поверхностной лексики, то "
             "есть превосходит ли дельта дешёвый базовый метод TF-IDF (§9, §11). Вопрос точности мы не снимали; но "
             "именно заранее зафиксированное правило ценности оказалось решающим по итогам.")
    para(doc, "Метафорически: активационная дельта — это микроскоп (доступ к скрытым состояниям, самостоятельно "
             "размещённая модель, реальные вычислительные затраты), а TF-IDF по самим словам — лупа (дёшево и "
             "мгновенно). Вопрос статьи: не берёмся ли мы за микроскоп там, где достаточно лупы. Как покажут "
             "результаты — для этой задачи лупа не просто достаточна, а в чём-то даже точнее микроскопа.")
    para(doc, "Основной практический фокус — ложные срабатывания на доброкачественных (benign) императивных "
             "ограничениях (например, «сделай выжимку без чисел»). Защитный механизм, который помечает (flag) такие "
             "инструкции как атаку, приводит к операционным сбоям. Мы проверяем, умеет ли зонд различать легитимный "
             "императив и попытку переопределения.")

    heading(doc, "Метод")
    para(doc, "Дельта определяется как Δ = pool(f(полный промпт)) − pool(f(системный промпт)); здесь системный промпт "
             "— только системная часть, а полный промпт — системный плюс пользовательский. Агрегация (pooling) — "
             "последний непустой токен на слоях {3,6,12}, конкатенация в 7680-мерный вектор, нормировка "
             "Δ_норм = Δ / ‖pool(f(системный промпт))‖. Зонд: StandardScaler → PCA(16) → логистическая регрессия с балансировкой классов. "
             "Модель заморожена, вычисления на CPU в формате bfloat16. Порог (threshold) τ = 0.258 выбран на валидации "
             "при полноте 0.95.")
    para(doc, "Обоснование гиперпараметров. Слои {3,6,12} выбраны как репрезентативный охват ранней, средней и поздней "
             "трети сети; раздел E2 отдельно проверяет, меняются ли выводы на других глубинах. Целевая полнота 0.95 для "
             "порога τ задана операционно: защитный механизм должен поймать 95% атак, пока доля ложных блокировок "
             "остаётся в допуске; τ = 0.258 — наименьшее значение, обеспечивающее эту полноту на валидации.")
    figure(doc, "figures/fig_method_ru.png",
           "Рис. 1. Схема метода. (а) общий вид двухпроходной дельты; (б) подробный конвейер обработки: два прохода "
           "через замороженную модель, отводы слоёв {3,6,12}, агрегация последнего токена, вычитание, нормировка, "
           "зонд и решение по порогу.", width_cm=15.0)
    para(doc, "Дизайн 2×2: пересечение признака переопределения (O) и вредоносности (harm, H) даёт ячейки C1 "
             "(O=1,H=1), C2 (O=0,H=1), C3 (O=1,H=0), C4 (O=0,H=0), плюс ОГРАНИЧЕНИЯ (CONSTRAINT) — доброкачественные "
             "императивные фильтры контента. Типы ограничений T1–T20: пропуск чисел, дат, личных имён, методологии, "
             "прямых цитат, URL, жаргона и мнений; сжатие до двух предложений; ограничение по названному объекту; без "
             "примеров, цитирований, кода, повторов и названий компаний; официальный регистр; без личных историй, "
             "шаблонных фраз и указания авторства; только сравнения. Разделение без утечек: обучение на семействах F1–F2, тест на F3–F10; "
             "системные промпты SP1–5 для обучения, SP6–10 для теста; типы ограничений также разделены. "
             "Статистический вывод — кластерный бутстрап по семействам, промптам и типам (B=2000). Значимость "
             "оценивается через кластерные бутстрап-доверительные интервалы и тесты эквивалентности (TOST), а не через "
             "p-значения: единица обобщения — семейство/промпт/тип, а не отдельная запись.")

    heading(doc, "Подтверждающие результаты")
    para(doc, "Формулируем проверяемые гипотезы. H1 (декодируемость по семействам): зонд различает override и "
             "не-override на удержанных семействах атак и системных промптах. H2 (различительная способность): "
             "направление считывания соответствует переопределению, а не вредоносности. H3 (избыточная защита): "
             "доброкачественные императивные ограничения не помечаются как override.")
    para(doc, "Объединённая площадь под ROC (переопределение против отсутствия переопределения) на тесте: 0.895. "
             "Гипотеза H1 (межсемейная, cross-family, обобщаемость) не подтверждена: средний AUROC по семействам "
             "0.877 (95% ДИ [0.828, 0.918]) ниже порога 0.90/0.85. Гипотеза H2 (различительная способность, "
             "discriminant validity: направление = переопределение, а не вредоносность) опровергнута: C3−C1 = +0.42, "
             "C2−C4 = +0.36, оба вне интервала эквивалентности ±0.10. Гипотеза H3 (доброкачественные ограничения не "
             "помечаются) формально подтверждена: частота ложных срабатываний 0.018 (95% ДИ [0.000, 0.054]) при "
             "полноте 0.90. Однако, как показано ниже согласованным тестом, это НЕ свидетельство специфичности зонда "
             "к переопределению: раз зонд не реагирует на переопределяющую обёртку, он не реагирует и на "
             "доброкачественные ограничения — устойчивость к избыточной защите неотличима от нечувствительности к "
             "самому сигналу.")
    caption_above(doc, "Таблица 1 – Override-ячейки: баллы и полнота (O=1)")
    add_table(doc,
        ["Ячейка", "n", "средний балл", "полнота@τ"],
        [["C1 переопределение+вредоносность", "400", "0.518", "0.840"],
         ["C3 переопределение+доброкач.", "250", "0.941", "1.000"]])
    caption_above(doc, "Таблица 2 – Не-override ячейки: баллы и ложные срабатывания (O=0)")
    add_table(doc,
        ["Ячейка", "n", "средний балл", "FPR@τ"],
        [["C2 без переопределения+вредоносность", "150", "0.617", "0.987"],
         ["C4 без переопределения+доброкач.", "150", "0.254", "0.327"],
         ["ОГРАНИЧЕНИЯ", "500", "0.044", "0.018"]])
    figure(doc, "figures/fig_percell_ru.png",
           "Рис. 2. Распределение баллов зонда по ячейкам. Вредоносный без переопределения (C2) получает такой же "
           "балл, как при переопределении — видно смешение по источнику.")

    heading(doc, "Базовые методы: добавляет ли дельта ценность")
    para(doc, "Ключевая проверка. Классификатор TF-IDF (граммы 1–2) на исходном (raw) пользовательском тексте даёт "
             "AUROC 0.929, что ВЫШЕ активационной дельты (0.895); разница −0.034. По заранее зафиксированному "
             "правилу, если разница меньше 0.05, внутренняя дельта не даёт существенного преимущества перед "
             "поверхностной лексикой. Более того, TF-IDF также почти не помечает доброкачественные ограничения "
             "(FPR на ОГРАНИЧЕНИЯХ = 0.000), то есть устойчивость к избыточной защите — свойство данных, а не метода. "
             "Контрольный запуск специализированного предобученного детектора DeBERTa (zero-shot) даёт AUROC 0.888 — "
             "тоже ниже TF-IDF: даже «навороченный микроскоп» не обгоняет лупу. Здесь DeBERTa — zero-shot ориентир "
             "на нашем распределении, без дообучения; 0.888 — не предел возможностей специализированного детектора, а "
             "точка сравнения.")
    caption_above(doc, "Таблица 3 – Сравнение базовых методов")
    add_table(doc,
        ["Модель", "Тест AUROC", "Δ относительно дельты"],
        [["TF-IDF (1–2 граммы) + лог.регрессия", "0.929", "+0.034"],
         ["Активационная дельта {3,6,12}", "0.895", "—"],
         ["DeBERTa-v3 (protectai, zero-shot)", "0.888", "−0.007"]])
    figure(doc, "figures/fig_roc_ru.png",
           "Рис. 3. ROC-кривые: переопределение против отсутствия переопределения на тесте. TF-IDF по исходному "
           "тексту (AUC 0.929) лежит выше активационной дельты (0.895).")

    heading(doc, "Декомпозиция 2×2 и смешение по источнику")
    para(doc, "Факторный анализ средних по ячейкам: главный эффект O = +0.294, главный эффект H = −0.031 "
             "(практически ноль), взаимодействие = −0.786. Важно: в нашем дизайне признак H коллинеарен источнику, "
             "а именно: C1 = «advbench_wrapped» (цели AdvBench в override-обёртке), C2 = «advbench» (прямые "
             "вредные запросы AdvBench), C3 = «authored» (наши авторские override-записи с доброкачественной "
             "нагрузкой), C4 = «alpaca» (инструкции Stanford Alpaca). Поэтому наблюдаемое «взаимодействие» — это "
             "фактически источник × переопределение, а не вредоносность × переопределение. Иными словами, "
             "переопределение поднимает балл на +0.69 для доброкачественного (авторского) контента, но на −0.10 для "
             "вредоносного (advbench) — различие объясняется источником, а не вредоносностью как таковой. "
             "Согласованный тест ниже снимает эту неоднозначность.")

    heading(doc, "Постфактумный анализ: согласованный по источнику тест H2 (на пяти системных промптах)")
    para(doc, "Этот анализ не входил в предрегистрацию и выполнен постфактум (post-hoc); он НЕ меняет формальные "
             "выводы H1–H3, но существенно уточняет их интерпретацию.")
    para(doc, "Для устранения смешения построены согласованные пары: одна и та же переопределяющая идиома, один и тот "
             "же системный промпт, согласованная длина полезной нагрузки (payload), меняется только вредоносность. "
             "Проверено на всех пяти тестовых системных промптах. AUROC «переопределение против прямого запроса» "
             "оказался на уровне случайности (chance level) для каждого промпта (0.461–0.487, объединённый 0.480, "
             "разброс 0.026). Эффект вредоносности снизился с +0.42 до ≈ +0.13 и согласован по промптам. Вывод: при "
             "фиксированном контенте зонд не реагирует на саму переопределяющую обёртку; исходная «сигнатура "
             "переопределения» шла от различий контента/источника.")
    caption_above(doc, "Таблица 4 – Согласованный по источнику тест H2 (пять системных промптов)")
    add_table(doc,
        ["Системный промпт", "Переопр.-против-прямого AUROC", "Вредоносность (переопр.)", "Вредоносность (прямой)"],
        [["SP6", "0.468", "+0.123", "+0.107"],
         ["SP7", "0.471", "+0.117", "+0.100"],
         ["SP8", "0.487", "+0.150", "+0.141"],
         ["SP9", "0.474", "+0.157", "+0.140"],
         ["SP10", "0.461", "+0.129", "+0.117"],
         ["Объединённо", "0.480", "+0.135", "+0.121"]])

    heading(doc, "Послойная декодируемость (E2)")
    para(doc, "Обучение зонда отдельно на каждом слое показывает, что более глубокие слои (12–20) декодируют лучше "
             "поверхностных, что согласуется с поздним слоем, где принимается решение. Однако ни один слой не "
             "достигает базового TF-IDF: лучший — слой 20 со значением 0.915. Обфускация кодированием (F5) остаётся "
             "слабой на всех глубинах (0.72–0.79).")
    caption_above(doc, "Таблица 5 – Послойная декодируемость (E2)")
    add_table(doc,
        ["Слой", "1", "4", "8", "12", "16", "20", "24", "28", "31"],
        [["Объединённый AUROC", ".880", ".889", ".876", ".907", ".906", ".915", ".893", ".899", ".908"],
         ["F5 (кодирование)", ".771", ".790", ".716", ".751", ".761", ".783", ".723", ".728", ".790"]])
    figure(doc, "figures/fig_e2_ru.png",
           "Рис. 4. E2: послойная декодируемость. Ни один слой не достигает уровня TF-IDF.")

    heading(doc, "Внешняя валидация на PIDS-Bench")
    para(doc, "На независимом тесте PIDS-Bench (MIT/CC-BY/Apache) взято опубликованное подмножество "
             "overdefense_test (339 сложных доброкачественных примеров), сопоставленных с тестовыми системными "
             "промптами. Средняя частота ложных срабатываний 0.018, наихудший случай 0.068 — далеко ниже порога "
             "0.20. Устойчивость к избыточной защите воспроизводится на сторонних данных, но, как показано выше, не "
             "является специфичной для активационной дельты.")

    heading(doc, "Разведочный двуязычный эксперимент")
    para(doc, "Добавление машинного перевода на русский в обучение поднимает полноту на русском тесте с 0.60 до 0.85 "
             "при почти нулевой частоте ложных срабатываний. Проверка обратного переноса: на английском тесте зонд, "
             "обученный с русскими данными, теряет лишь 0.011 AUROC. Многоязычность ограничена покрытием данными, а "
             "не пределом метода. Эксперимент не был предрегистрирован и представлен как разведочный.")

    heading(doc, "Проверка на масштабе модели (4B → 9B)")
    para(doc, "Чтобы проверить, не является ли негативный результат артефактом малой модели, мы повторили конвейер "
             "на Qwen3.5-9B (расширение после предрегистрации). Увеличение параметров дало лишь прирост на уровне "
             "погрешности и не изменило ни одного вывода: объединённый AUROC дельты вырос с 0.895 до 0.907 (+0.012), "
             "но средний по семействам остался ниже порога (0.890 < 0.90); взаимодействие в H2 не изменилось "
             "(+0.42 / +0.33); и, главное, согласованный тест по-прежнему на уровне случайности (0.48 → 0.46) — "
             "большая модель НЕ начинает видеть переопределяющую обёртку. TF-IDF (0.929) остаётся выше 9B-дельты. "
             "Вывод: негативный результат устойчив к масштабу; «больше — не значит лучше» для данной задачи.")
    caption_above(doc, "Таблица 6 – Масштаб модели 4B → 9B")
    add_table(doc,
        ["Метрика", "Qwen3.5-4B", "Qwen3.5-9B"],
        [["Объединённый AUROC (дельта)", "0.895", "0.907"],
         ["H1 среднее по семействам", "0.877", "0.890"],
         ["H2 C3−C1", "+0.42", "+0.42"],
         ["H2 C2−C4", "+0.36", "+0.33"],
         ["Согласованный H2 (обёртка)", "0.480", "0.459"],
         ["H3 FPR / полнота", "0.018 / 0.902", "0.000 / 0.872"],
         ["TF-IDF (от модели не зависит)", "0.929", "0.929"]])
    heading(doc, "Отклонения от предрегистрации")
    para(doc, "Применено предписанное §4 переопределение конструкта с «обнаружение переопределения» на «обнаружение "
             "вредоносных/переопределяющих запросов». Для TOST использован 90% ДИ (стандарт эквивалентности) вместо "
             "упомянутых в тексте 95%. Фактический τ = 0.258 вместо иллюстративных 0.40. Двуязычный эксперимент и "
             "расширенный согласованный тест H2 выполнены после раскрытия тестовой выборки и помечены как разведочные; "
             "базовые методы и E2 были предрегистрированы (§11, §15) и не меняют подтверждающий вывод.")
    para(doc, "По базовым методам §11: выполнен TF-IDF (см. выше). Реализация линейного зонда на активациях в духе "
             "TaskTracker [1] по сути совпадает с нашим дельта-зондом, поэтому отдельным базовым методом не "
             "дублируется. Классификатор DeBERTa (protectai) прогнан как zero-shot базовый метод (см. выше). "
             "Контрольный эксперимент с согласованием по длине и теме не выполнен из-за ограничений по CPU; это "
             "осознанное ограничение, отмеченное для полноты. Тем не менее постфактумный source-matched анализ "
             "(Таблица 4) частично компенсирует этот пробел, показывая тот же эффект на уровне случайности.")
    para(doc, "Проверка масштаба модели (4B → 9B) выполнена после предрегистрации и раскрытия тестовой выборки; она "
             "не входила в план и помечена как разведочная. Результат — прирост на уровне погрешности, ни один вывод "
             "не меняется.")

    heading(doc, "Обсуждение")
    para(doc, "Почему согласованный тест даёт уровень случайности. Возможные объяснения: (i) выбранные слои {3,6,12} "
             "могут быть слишком ранними для инструкции; (ii) агрегация последнего токена при длине 256 способна "
             "терять сигнал, если информация об инструкции распределена по последовательности; (iii) сама конструкция "
             "дельты pool(полный промпт) − pool(системный промпт) может подавлять именно компонент, связанный с "
             "инструкцией, оставляя лишь общий сдвиг представления от добавления любого текста. Различимые по контенту "
             "ячейки (разные источники) дают зонду поверхностные признаки, которых хватает на исходном тесте, но "
             "исчезают при фиксации контента. В будущих работах стоит проверить альтернативные схемы агрегации — "
             "например, разность активаций по токенам-триггерам или attention-weighted pooling, — чтобы окончательно "
             "установить, не декодируется ли переопределение в принципе или наблюдаемая слепота является артефактом "
             "конкретной схемы пулинга.")
    para(doc, "Связь с [7] (Wen et al.). В [7] ранние слои декодируемы, но обфускация их ломает, а причинное «узкое "
             "место» (bottleneck) лежит в поздних слоях. Наш E2 согласуется с тем, что поздние слои (12–20) "
             "декодируют лучше ранних, однако ни один не достигает TF-IDF. Вероятно, результат [7] получен на другой "
             "конструкции — абсолютных активациях, а не разностной дельте; прямое сопоставление требует отдельной "
             "проверки.")
    para(doc, "Практический вывод для систем защиты. Если TF-IDF на исходном тексте даёт AUROC 0.929 и почти "
             "нулевые ложные срабатывания на доброкачественных ограничениях, то для практического защитного "
             "механизма дешёвый текстовый базовый метод — более разумный выбор, чем активационная дельта, требующая "
             "доступа к скрытым состояниям и локально размещённой модели.")

    heading(doc, "Ограничения исследования")
    para(doc, "Декодируемость ≠ причинность ≠ защита: измеряется линейная считываемость коррелята, а не причинное "
             "влияние и не сквозная защищённость (причинный патчинг отложен, E6). Выводы сделаны для одной модели "
             "(Qwen3.5, 4B/9B) и одной схемы агрегации (последний токен); другие модели и схемы (trigger-token, "
             "attention-weighted) могут дать иной результат.")
    para(doc, "Конструкт «переопределение» частично перекрывается с контентом/источником: source-matched анализ это "
             "показывает, но полностью развести их на исходных данных нельзя. H1 на грани — 0.877 (4B) / 0.890 (9B) "
             "против порога 0.90, то есть близкий промах, а не нулевой результат.")
    para(doc, "Часть бейзлайнов §11 (length/topic-matched контроль) не выполнена из-за CPU-ограничений; DeBERTa — "
             "zero-shot ориентир без дообучения. Двуязычный эксперимент и проверка масштаба выполнены post-hoc и не "
             "были предрегистрированы.")

    heading(doc, "Заключение")
    para(doc, "Результат строгий и в целом негативный. Для данной задачи активационная дельта не превосходит TF-IDF "
             "по исходному тексту, не реагирует на переопределяющую обёртку при контроле контента, а её кажущаяся "
             "«сигнатура переопределения» объясняется различиями источника. Единственный устойчивый (robust) "
             "результат — устойчивость к избыточной защите — оказывается свойством распределения данных, а не метода. "
             "Это полезное ограничение семейства методов: оно предостерегает от заявлений об «обнаружении "
             "переопределения» на основе дельты, которую превосходит простой текстовый базовый метод.")

    heading(doc, "Доступность данных и кода")
    para(doc, "Полный воспроизводимый пакет — замороженный датасет, скрипты пайплайна (T0–T5 и проверка масштаба), "
             "рисунки, отчёты и рукопись — доступен в репозитории: "
             "https://github.com/in40/10.17605-OSF.IO-EP6AB (CC0). Предрегистрация: OSF ep6ab, "
             "DOI 10.17605/OSF.IO/EP6AB.")

    heading(doc, "Глоссарий")
    add_table(doc,
        ["Термин", "Определение"],
        [["prompt-инъекция (prompt injection)", "атака: пользовательский ход пытается отменить или подменить системный промпт"],
         ["системный промпт (system prompt)", "инструкция, задающая правила поведения модели"],
         ["полный промпт (full prompt)", "системный промпт плюс пользовательский ход"],
         ["активационная дельта (activation delta)", "разность пулов скрытых состояний: pool(полный) − pool(системный)"],
         ["зонд (probe)", "линейный классификатор, обученный поверх дельты"],
         ["TF-IDF", "базовый метод: вес слова = частота в тексте × обратная частота по корпусу; классификатор по «мешку слов»"],
         ["агрегация (pooling)", "сведение последовательности токенов к одному вектору (последний непустой токен)"],
         ["нормировка", "деление дельты на норму базовой дельты"],
         ["порог τ (threshold)", "значение скора, выше которого запись помечается как override"],
         ["полнота (recall)", "доля истинных override, корректно пойманных"],
         ["FPR", "доля легитимных записей, ложно помеченных как override"],
         ["ДИ (доверительный интервал)", "интервал, накрывающий параметр с вероятностью 95%"],
         ["кластерный бутстрап", "оценка ДИ повторной выборкой целых кластеров (семейств/промптов/типов)"],
         ["эквивалентность (TOST)", "тест «два односторонних»: различие внутри допуска ±0.10"],
         ["семейство атак (attack family)", "класс по механизму переопределения (F1–F10)"],
         ["тип ограничения (constraint type)", "вид доброкачественного императивного фильтра (T1–T20)"],
         ["обфускация", "маскировка атаки кодированием или искажением"],
         ["избыточная защита (over-defense)", "ложные блокировки легитимных запросов"],
         ["защитный механизм (guardrail)", "фильтр, отсекающий вредоносные запросы"],
         ["переопределение (override)", "намерение отменить системные инструкции"],
         ["доброкачественный (benign)", "легитимный, не несущий вреда"],
         ["вредоносность (harm)", "свойство запроса нести вред"]])

    heading(doc, "Литература")
    refs = [
        "Abdelnabi S., Fay B., Cherubin A., Salem A., Fritz M., Paverd A. Get my drift? Catching LLM task drift with activation deltas // IEEE SaTML. 2025. arXiv:2406.00799. DOI:10.1109/SaTML64287.2025.00011.",
        "Liu Y., Cheng W., Zheng L., Su Z. Defending LLMs against jailbreak attacks through representation offset detection // Information Processing & Management. 2026. Vol. 63(5). 104662. DOI:10.1016/j.ipm.2026.104662.",
        "Ye S., Cui H., Hadfield-Menell D. Prompt injection as role confusion // ICML. 2026. arXiv:2603.12277.",
        "Guo W., Zhao J., Wang B. BERM: Low-overhead prompt-injection detection via in-situ benign representation modeling // IJCAI. 2026. P. 599–607. DOI:10.24963/ijcai.2026/68.",
        "Wang Y., Wei A., Liu Z., Chen X. False sense of security: Why probing-based malicious input detection fails to generalize // NeurIPS 2025 MechInterp WS. arXiv:2509.03888.",
        "David R. et al. Latent undertow: Representation dynamics under injection // ICML 2026 MechInterp. arXiv:2609.15994.",
        "Wen J. et al. Where do LLMs decide? // arXiv:2609.37737.",
        "Fay B. et al. Persistent homology of LLM representations // ICLR. 2026. arXiv:2505.20435.",
        "Shire. PIDS-Bench: Evaluating prompt-injection detectors under over-defense, obfuscation, and distribution shift // arXiv:2609.15017.",
        "Dong et al. I've decided to leak // EMNLP. 2025. 2025.emnlp-main.1082.",
        "Russinovich. CounterSteer // arXiv:2609.36570.",
        "Greshake K. et al. Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection // arXiv:2302.12173.",
    ]
    for i, r in enumerate(refs, 1):
        para(doc, f"{i}. {r}", size=10, align='left', indent=False, space_after=2)

    heading(doc, "Об авторе")
    para(doc, "Сорокин Александр — независимый исследователь в области безопасности LLM, машинной интерпретируемости "
             "и предрегистрированных эмпирических исследований. Работа выполнена самостоятельно, в нерабочее время, без использования "
             "ресурсов и данных работодателя и без внешнего финансирования. Контакт: sorokin@sorokinonline.com.",
         size=10, align='left', indent=False)

    heading(doc, "Заявление об использовании искусственного интеллекта")
    para(doc, "При подготовке данной работы использовались большие языковые модели (ИИ-ассистент): для помощи в "
             "формулировании и редактировании текста, генерации и отладки кода анализа, построения рисунков и "
             "проверки оформления. Машинный перевод (opus-mt-en-ru) применялся для русскоязычной ветки данных (что "
             "отмечено в ограничениях). Исследуемые модели (Qwen3.5-4B/9B) являются объектом исследования, а не "
             "инструментом формулирования выводов. Все сгенерированные материалы проверены, отредактированы и "
             "подтверждены автором; ответственность за содержание, корректность цитат и научную добросовестность "
             "полностью несёт автор. Раскрытие выполнено в соответствии с политикой журнала об использовании "
             "генеративного ИИ.")

# ============================================================
# ENGLISH VERSION
# ============================================================
def english(doc):
    global _LANG
    _LANG = "en-US"
    para(doc, "Activation-Delta Probes Compared with TF-IDF for Prompt-Injection Detection: A Pre-Registered Study",
         size=12, bold=True, align='left', indent=False, space_after=6)
    para(doc, "Aleksandr Sorokin", align='left', indent=False, space_after=2)
    para(doc, "Independent Researcher", align='left', indent=False, space_after=2)
    para(doc, "sorokin@sorokinonline.com", align='left', indent=False, space_after=2)
    para(doc, "No external funding was received. The work was performed independently, without the use of any "
              "employer resources or data.", size=9, italic=True, align='left', indent=False, space_after=8)

    heading(doc, "Abstract")
    para(doc, "A pre-registered study (OSF ep6ab) of what the two-pass activation delta of a frozen LLM (Qwen3.5-4B) "
             "reads during an attempted prompt-injection. The method is not novel; the contribution is a strict "
             "leak-free evaluation with cluster-level inference. Three key results. First, a source- and length-matched "
             "test shows that with content held fixed the probe does not respond to the override wrapper itself "
             "(AUROC ≈ 0.48 across all five held-out prompts). Second, a TF-IDF baseline outperforms the delta "
             "(0.929 vs 0.895). Third, the apparent over-defense resistance (low false positives on benign "
             "constraints) is a property of the data, not the method. The delta encodes content and source "
             "differences, not override intent; for a guardrail, a cheap text baseline is the sensible choice.")

    heading(doc, "Keywords")
    para(doc, "prompt injection; activation deltas; linear probing; over-defense; false positives; pre-registration; "
             "frozen LLM; TF-IDF.", align='left', indent=False)

    heading(doc, "Introduction")
    para(doc, "Prompt-injection attacks attempt to make a user turn override or replace a large language model's system "
             "prompt. The activation-delta family of detectors assumes that this override intent is linearly decodable "
             "from the hidden states of a frozen model. We pre-registered (OSF ep6ab, DOI 10.17605/OSF.IO/EP6AB) two "
             "questions. The first is decodability: is the override intent linearly readable from the delta "
             "(hypotheses H1–H3)? The second is value: does the internal state add anything beyond surface lexical "
             "features — that is, does the delta beat a cheap TF-IDF baseline (§9, §11)? We did not drop the accuracy "
             "question; but the pre-registered value rule turned out decisive.")
    para(doc, "Metaphorically, the activation delta is a microscope — access to hidden states, a self-hosted model, "
             "real compute — while TF-IDF on the words themselves is a magnifying glass: cheap and instant. The "
             "question this paper asks is whether we are reaching for a microscope where a magnifying glass would do. "
             "As the results show, for this task the magnifying glass is not merely sufficient but, in some respects, "
             "sharper than the microscope.")
    para(doc, "The practical focus is over-defense [9]: false positives on benign imperative constraints (e.g., "
             "\"summarize without the numbers\"). A guardrail that flags such instructions causes operational "
             "disruptions. We test whether the probe distinguishes a legitimate imperative from an override attempt.")

    heading(doc, "Method")
    para(doc, "The delta is Δ = pool(f(full prompt)) − pool(f(system prompt)); here the system prompt is the system part "
             "alone, and the full prompt is the system plus the user prompt. Pooling is the last non-padding token at "
             "layers {3,6,12}, concatenated to 7680 dimensions, normalized as Δ_norm = Δ / ‖pool(f(system prompt))‖. "
             "The probe is "
             "StandardScaler → PCA(16) → balanced logistic regression; we refer to the linear classifier on the delta "
             "as the probe (or delta probe) interchangeably. The model is frozen; computation is on CPU in bfloat16. "
             "The threshold τ = 0.258 is chosen on validation at a recall of 0.95.")
    para(doc, "Hyperparameter rationale. Layers {3,6,12} are chosen as a representative early/middle/late-third "
             "coverage of the network; section E2 separately checks whether conclusions change at other depths. The "
             "target recall of 0.95 for the threshold is set operationally: a guardrail should catch 95% of attacks "
             "while the false-block rate stays within tolerance; τ = 0.258 is the smallest value achieving that recall "
             "on validation.")
    figure(doc, "figures/fig_method.png",
           "Fig. 1. Method scheme. (a) overview of the two-pass delta; (b) detailed pipeline: two passes through the "
           "frozen model, layer taps {3,6,12}, last-token pooling, subtraction, normalization, probe, and the "
           "threshold decision.", width_cm=15.0)
    para(doc, "The design is a 2×2 crossing of override (O) and harm (H): C1 (O=1,H=1), C2 (O=0,H=1), C3 (O=1,H=0), "
             "C4 (O=0,H=0), plus CONSTRAINT (benign imperative content filters). Constraint types T1–T20: skip "
             "numbers, dates, personal names, methodology, direct quotes, URLs, jargon and opinions; compress to two "
             "sentences; scope to a named object; no examples, citations, code, repetition or company names; formal "
             "register; no personal stories, boilerplate or attribution; comparisons only. Leak-free split: train on families "
             "F1–F2, test on F3–F10; system prompts SP1–5 for training, SP6–10 for test; constraint types also split. "
             "Statistical inference uses a cluster bootstrap over families, prompts, and types (B=2000). Significance "
             "is assessed via cluster bootstrap confidence intervals and equivalence tests (TOST), not p-values: the "
             "unit of generalization is the family/prompt/type, not the individual record.")

    heading(doc, "Confirmatory results")
    para(doc, "We state the testable hypotheses. H1 (cross-family decodability): the probe distinguishes override from "
             "non-override on held-out attack families and system prompts. H2 (discriminant validity): the read-off "
             "direction corresponds to override, not to harmfulness. H3 (over-defense): benign imperative constraints "
             "are not flagged as override.")
    para(doc, "Pooled test AUROC (override vs non-override) is 0.895. H1 (cross-family generalization) is not met: "
             "mean per-family AUROC 0.877 (95% CI [0.828, 0.918]), below the 0.90/0.85 threshold. H2 (discriminant "
             "validity: direction = override, not harm) is refuted: C3−C1 = +0.42 and C2−C4 = +0.36, both outside the "
             "±0.10 equivalence margin. H3 (benign constraints not flagged) is formally confirmed: false-positive rate "
             "0.018 (95% CI [0.000, 0.054]) at recall 0.90. However, as the matched analysis below shows, this is not "
             "evidence of override-specificity: since the probe does not respond to the override wrapper, it does not "
             "respond to benign constraints either — over-defense resistance is indistinguishable from insensitivity "
             "to the signal itself.")
    caption_above(doc, "Table 1. Override cells: scores and recall (O=1)")
    add_table(doc,
        ["Cell", "n", "mean score", "recall@τ"],
        [["C1 override+harm", "400", "0.518", "0.840"],
         ["C3 override+benign", "250", "0.941", "1.000"]])
    caption_above(doc, "Table 2. Non-override cells: scores and false positives (O=0)")
    add_table(doc,
        ["Cell", "n", "mean score", "FPR@τ"],
        [["C2 no-override+harm", "150", "0.617", "0.987"],
         ["C4 no-override+benign", "150", "0.254", "0.327"],
         ["CONSTRAINT", "500", "0.044", "0.018"]])
    figure(doc, "figures/fig_percell.png",
           "Fig. 2. Per-cell probe-score distribution. Harmful non-override (C2) scores like override — "
           "the source confound is visible.")

    heading(doc, "Baselines: Does the Delta Add Value?")
    para(doc, "This is the key test. A TF-IDF (1–2-gram) classifier on raw user text reaches an AUROC of 0.929, higher "
             "than the activation delta (0.895); the gap is −0.034. By the pre-registered rule, a gap below 0.05 means "
             "the internal-state delta adds no substantial value over surface text. Moreover, TF-IDF also rarely flags "
             "benign constraints (CONSTRAINT FPR = 0.000), so over-defense resistance is a property of the data, not "
             "of the method. A zero-shot run of the specialized pretrained DeBERTa detector reaches AUROC 0.888 — "
             "also below TF-IDF: even a fancier microscope does not beat the magnifying glass. Here DeBERTa is a "
             "zero-shot reference on our distribution, not fine-tuned; 0.888 is not the ceiling of a specialized "
             "detector but a comparison point.")
    caption_above(doc, "Table 3. Baseline comparison")
    add_table(doc,
        ["Model", "Test AUROC", "Δ vs delta"],
        [["TF-IDF (1–2-gram) + logistic regression", "0.929", "+0.034"],
         ["Activation delta {3,6,12}", "0.895", "—"],
         ["DeBERTa-v3 (protectai, zero-shot)", "0.888", "−0.007"]])
    figure(doc, "figures/fig_roc.png",
           "Fig. 3. ROC curves for override vs non-override on the test set. TF-IDF on raw user text "
           "(AUC 0.929) lies above the activation delta (0.895).")

    heading(doc, "2×2 decomposition and the source confound")
    para(doc, "A factorial analysis of cell means shows: main effect of O = +0.294, main effect of H = −0.031 (≈ 0), "
             "interaction = −0.786. Crucially, in our design the harm factor H is collinear with source — namely "
             "C1 = \"advbench_wrapped\" (AdvBench goals wrapped in an override idiom), C2 = \"advbench\" (direct "
             "harmful AdvBench requests), C3 = \"authored\" (our authored override records with a benign payload), "
             "C4 = \"alpaca\" (Stanford Alpaca instructions). So the observed \"interaction\" is actually source × "
             "override, not harm × override. That is, override raises the score by +0.69 on benign (authored) content "
             "but by −0.10 on harmful (advbench) content — the difference is explained by source, not by harm per se. "
             "The matched test below removes this ambiguity.")

    heading(doc, "Post-hoc analysis: source-matched H2 (across five system prompts)")
    para(doc, "This analysis was not part of the pre-registration and is performed post hoc; it does not change the "
             "formal H1–H3 verdicts but substantially refines their interpretation.")
    para(doc, "To remove the confound we built matched pairs: the same override idiom, the same system prompt, matched "
             "payload length, varying only harm. When run across all five held-out system prompts, the override-vs-direct "
             "AUROC is at chance level for every prompt (0.461–0.487, pooled 0.480, spread 0.026). The harm effect "
             "collapses from +0.42 to ≈ +0.13 and is consistent across prompts. With content held fixed, the probe does "
             "not respond to the override wrapper; the apparent override signature came from content/source differences.")
    caption_above(doc, "Table 4. Source-matched H2 (five system prompts)")
    add_table(doc,
        ["System prompt", "Ov-vs-dir AUROC", "Harm (ov)", "Harm (dir)"],
        [["SP6", "0.468", "+0.123", "+0.107"],
         ["SP7", "0.471", "+0.117", "+0.100"],
         ["SP8", "0.487", "+0.150", "+0.141"],
         ["SP9", "0.474", "+0.157", "+0.140"],
         ["SP10", "0.461", "+0.129", "+0.117"],
         ["Pooled", "0.480", "+0.135", "+0.121"]])

    heading(doc, "Layer-wise decodability (E2)")
    para(doc, "Training a probe on each layer individually shows that deeper layers (12–20) decode better than shallow "
             "ones, consistent with a decision locus in later layers. Yet no layer reaches the TF-IDF baseline: the best "
             "is layer 20 at 0.915. Encoding/obfuscation (F5) remains weak at every depth (0.72–0.79).")
    caption_above(doc, "Table 5. Layer-wise decodability (E2)")
    add_table(doc,
        ["Layer", "1", "4", "8", "12", "16", "20", "24", "28", "31"],
        [["Pooled AUROC", ".880", ".889", ".876", ".907", ".906", ".915", ".893", ".899", ".908"],
         ["F5 (encoding)", ".771", ".790", ".716", ".751", ".761", ".783", ".723", ".728", ".790"]])
    figure(doc, "figures/fig_e2.png",
           "Fig. 4. E2: layer-wise decodability. No layer reaches the TF-IDF baseline.")

    heading(doc, "External validation on PIDS-Bench")
    para(doc, "On the independent PIDS-Bench (MIT/CC-BY/Apache) we use the published overdefense_test subset (339 "
             "hard-benign (security-adjacent but legitimate) items) paired with held-out system prompts. Mean "
             "false-positive rate is 0.018, worst-case 0.068, far below the 0.20 threshold. Over-defense resistance is "
             "replicated on third-party data but, as shown above, is not specific to the activation delta.")

    heading(doc, "Exploratory bilingual experiment")
    para(doc, "Adding machine-translated Russian to training raises Russian-test recall from 0.60 to 0.85 with "
             "near-zero false positives. The negative-transfer check shows the Russian-augmented probe loses only 0.011 "
             "AUROC on the English test. Multilingual performance is limited by coverage, not a limitation of the "
             "method. This experiment was not pre-registered and is reported as exploratory.")

    heading(doc, "Model-scale robustness check (4B → 9B)")
    para(doc, "To test whether the negative result is an artifact of a small model, we repeated the pipeline on "
             "Qwen3.5-9B (a post-registration extension). Increasing parameters gave only a gain at the noise level and "
             "changed no conclusion: the pooled delta AUROC rose from 0.895 to 0.907 (+0.012), but the mean per-family "
             "AUROC stayed below the bar (0.890 < 0.90); the H2 interaction was unchanged (+0.42 / +0.33); and, "
             "crucially, the matched test remained at chance (0.48 → 0.46) — the larger model does NOT begin to see the "
             "override wrapper. TF-IDF (0.929) still exceeds the 9B delta. Conclusion: the negative result is robust "
             "to model scale; bigger is not better for this task.")
    caption_above(doc, "Table 6. Model scale 4B → 9B")
    add_table(doc,
        ["Metric", "Qwen3.5-4B", "Qwen3.5-9B"],
        [["Pooled AUROC (delta)", "0.895", "0.907"],
         ["H1 mean per-family", "0.877", "0.890"],
         ["H2 C3−C1", "+0.42", "+0.42"],
         ["H2 C2−C4", "+0.36", "+0.33"],
         ["Matched H2 (wrapper)", "0.480", "0.459"],
         ["H3 FPR / recall", "0.018 / 0.902", "0.000 / 0.872"],
         ["TF-IDF (model-independent)", "0.929", "0.929"]])

    heading(doc, "Deviations from the pre-registration")
    para(doc, "We applied the §4-mandated re-scope from \"override detection\" to \"detection of harmful/override-related "
             "prompts\". We used a 90% CI for TOST (equivalence standard) instead of the 95% mentioned in the text. The "
             "actual τ = 0.258 replaces the illustrative 0.40. The bilingual and extended source-matched H2 were run "
             "after unblinding and are labeled exploratory; the baselines and E2 were pre-registered (§11, §15) and do "
             "not change the confirmatory verdict.")
    para(doc, "On the §11 baselines: TF-IDF was implemented and evaluated (above). A TaskTracker-style linear probe on "
             "activations [1] is essentially our delta probe and is not duplicated as a separate baseline. The DeBERTa "
             "injection classifier (protectai) was run as a zero-shot baseline (above). The length/topic-matched "
             "control was not run due to CPU constraints; this is an acknowledged limitation noted for completeness. "
             "Nevertheless, the post-hoc source-matched analysis (Table 4) partially compensates for this gap, showing "
             "the same chance-level effect.")
    para(doc, "The model-scale check (4B → 9B) was performed after pre-registration and after unblinding the test "
             "split; it was not part of the plan and is labeled exploratory. The result is a gain at the noise level, "
             "and no conclusion changes.")

    heading(doc, "Discussion")
    para(doc, "Why the matched test yields chance-level performance. Possible explanations: (i) the chosen layers "
             "{3,6,12} may be too early to capture the instruction; (ii) last-token pooling at a length of 256 may lose "
             "signal if the instruction is spread across the sequence; (iii) the delta construction "
             "pool(full prompt) − pool(system prompt) may itself suppress the instructional component, leaving only a "
             "generic representation shift from adding any text. Cells that differ in content (different sources) give "
             "the probe surface features that suffice on the original test but vanish when content is held fixed. "
             "Future work should test alternative pooling schemes — e.g., differencing activations at trigger tokens or "
             "attention-weighted pooling — to settle whether override is undecodable in principle or the observed "
             "blindness is an artifact of this particular aggregation.")
    para(doc, "Relation to [7] (Wen et al.). [7] finds early layers decodable but degraded by obfuscation, with a "
             "causal bottleneck in later layers. Our E2 agrees that later layers (12–20) decode better than early ones, "
             "yet none reaches TF-IDF. The [7] result was likely obtained using a different construction — absolute "
             "activations rather than the difference delta; a direct comparison requires a separate test.")
    para(doc, "Practical implication for guardrails. If TF-IDF on raw user text reaches AUROC 0.929 with near-zero "
             "false positives on benign constraints, then for a practical guardrail a cheap text baseline is the more "
             "sensible choice than an activation delta that requires access to hidden states and a self-hosted model. "
             "The delta is justified only where a signal unavailable from text is required.")

    heading(doc, "Limitations of the study")
    para(doc, "Decodability ≠ causality ≠ defense: we measure the linear readability of a correlate, not causal "
             "leverage or end-to-end protection (causal patching is deferred, E6). Conclusions are drawn for a single "
             "model (Qwen3.5, 4B/9B) and a single aggregation scheme (last-token); other models and schemes "
             "(trigger-token, attention-weighted) may differ.")
    para(doc, "The override construct partly overlaps with content/source: the source-matched analysis shows this, but "
             "they cannot be fully separated in the original data. H1 is a near-miss — 0.877 (4B) / 0.890 (9B) "
             "against the 0.90 bar, i.e. a close miss rather than a null result.")
    para(doc, "Part of the §11 baselines (the length/topic-matched control) was not run due to CPU constraints; "
             "DeBERTa is a zero-shot reference without fine-tuning. The bilingual experiment and the model-scale check "
             "were performed post hoc and were not pre-registered.")

    heading(doc, "Conclusion")
    para(doc, "Our results are rigorous and largely negative. For this task the activation delta does not exceed a "
             "TF-IDF baseline on raw user text, does not respond to the override wrapper when content is controlled, "
             "and its apparent override signature is explained by source differences. The one stable finding, "
             "over-defense resistance, is a property of the data distribution rather than the method. This is a useful "
             "boundary for the method family: it cautions against claiming override detection from a delta that a simple "
             "text baseline matches.")

    heading(doc, "Code and Data Availability")
    para(doc, "The complete reproducible package — frozen dataset, pipeline scripts (T0–T5 and the model-scale check), "
             "figures, reports, and the manuscript — is available at "
             "https://github.com/in40/10.17605-OSF.IO-EP6AB (CC0). Pre-registration: OSF ep6ab, "
             "DOI 10.17605/OSF.IO/EP6AB.")

    heading(doc, "Glossary")
    add_table(doc,
        ["Term", "Definition"],
        [["prompt injection", "an attack where a user turn tries to override or replace the system prompt"],
         ["system prompt", "the instruction setting the model's behavioral rules"],
         ["full prompt", "the system prompt plus the user turn"],
         ["activation delta", "the difference of pooled hidden states: pool(full) − pool(system)"],
         ["probe", "a linear classifier trained on the delta"],
         ["TF-IDF", "baseline: word weight = term frequency × inverse document frequency; a bag-of-words classifier"],
         ["pooling", "reducing a token sequence to one vector (the last non-padding token)"],
         ["normalization", "dividing the delta by the baseline norm"],
         ["threshold τ", "the score above which a record is flagged as override"],
         ["recall", "fraction of true overrides correctly caught"],
         ["FPR", "fraction of legitimate records wrongly flagged as override"],
         ["CI (confidence interval)", "interval covering the parameter with 95% probability"],
         ["cluster bootstrap", "CI estimation by resampling whole clusters (families/prompts/types)"],
         ["equivalence (TOST)", "two-one-sided test: difference within the ±0.10 margin"],
         ["attack family", "class by override mechanism (F1–F10)"],
         ["constraint type", "kind of benign imperative filter (T1–T20)"],
         ["obfuscation", "masking an attack via encoding or distortion"],
         ["over-defense", "false blocking of legitimate requests"],
         ["guardrail", "a filter that blocks harmful requests"],
         ["override", "the intent to cancel system instructions"],
         ["benign", "legitimate, non-harmful"],
         ["harm", "the property of a request being harmful"]])

    heading(doc, "References")
    refs_en = [
        "Abdelnabi S., Fay B., Cherubin A., Salem A., Fritz M., Paverd A. Get my drift? Catching LLM task drift with activation deltas // IEEE SaTML. 2025. arXiv:2406.00799. DOI:10.1109/SaTML64287.2025.00011.",
        "Liu Y., Cheng W., Zheng L., Su Z. Defending LLMs against jailbreak attacks through representation offset detection // Information Processing & Management. 2026. Vol. 63(5). 104662. DOI:10.1016/j.ipm.2026.104662.",
        "Ye S., Cui H., Hadfield-Menell D. Prompt injection as role confusion // ICML. 2026. arXiv:2603.12277.",
        "Guo W., Zhao J., Wang B. BERM: Low-overhead prompt-injection detection via in-situ benign representation modeling // IJCAI. 2026. P. 599–607. DOI:10.24963/ijcai.2026/68.",
        "Wang Y., Wei A., Liu Z., Chen X. False sense of security: Why probing-based malicious input detection fails to generalize // NeurIPS 2025 MechInterp WS. arXiv:2509.03888.",
        "David R. et al. Latent undertow: Representation dynamics under injection // ICML 2026 MechInterp. arXiv:2609.15994.",
        "Wen J. et al. Where do LLMs decide? // arXiv:2609.37737.",
        "Fay B. et al. Persistent homology of LLM representations // ICLR. 2026. arXiv:2505.20435.",
        "Shire. PIDS-Bench: Evaluating prompt-injection detectors under over-defense, obfuscation, and distribution shift // arXiv:2609.15017.",
        "Dong et al. I've decided to leak // EMNLP. 2025. 2025.emnlp-main.1082.",
        "Russinovich. CounterSteer // arXiv:2609.36570.",
        "Greshake K. et al. Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection // arXiv:2302.12173.",
    ]
    for i, r in enumerate(refs_en, 1):
        para(doc, f"{i}. {r}", size=10, align='left', indent=False, space_after=2)

    heading(doc, "About the author")
    para(doc, "Aleksandr Sorokin is an independent researcher in LLM security, mechanistic interpretability, and "
             "pre-registered empirical research. The work was carried out in personal time, without the use of any "
             "employer resources or data and without external funding. Contact: sorokin@sorokinonline.com.",
         size=10, align='left', indent=False)

    heading(doc, "Declaration of Generative AI Use")
    para(doc, "Large language models (an AI assistant) were used during the preparation of this work to help draft and "
             "edit the text, generate and debug the analysis code, and produce figures. Machine translation "
             "(opus-mt-en-ru) was used for the Russian data arm (noted in the limitations). The models under study "
             "(Qwen3.5-4B/9B) are the object of research, not a tool for formulating conclusions. All AI-generated "
             "material was reviewed, edited, and verified by the author, who takes full responsibility for the "
             "content, the accuracy of citations, and the scientific integrity of the work. This disclosure follows "
             "the journal's policy on the use of generative AI.")


def main():
    doc = Document()
    setup(doc)
    russian(doc)
    doc.add_page_break()
    english(doc)
    out = "manuscript/manuscript_EP6AB.docx"
    doc.save(out)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
