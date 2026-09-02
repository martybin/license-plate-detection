# سامانه تشخیص پلاک ایرانی — دروازه معدن

تشخیص بلادرنگ پلاک کامیون‌های معدن از تصویر دوربین، و نمایش مشخصات راننده روی مانیتور دروازه.

سامانه از دو مدل تشکیل شده:

| مرحله | مدل | وظیفه | داده آموزش |
|---|---|---|---|
| ۱. تشخیص محل پلاک | YOLOv8 | پیدا کردن پلاک در فریم کامل دوربین | تصاویر **صحنه کامل خودرو** |
| ۲. خواندن متن پلاک | ResNet + BiLSTM + CTC | تبدیل برش پلاک به متن | تصاویر **برش پلاک** |

```
دوربین (RTSP)
    ↓  FrameGrabber  ← ترد جدا، حذف فریم‌های کهنه، اتصال مجدد خودکار
   فریم
    ↓  PlateDetector (YOLOv8)
  برش پلاک  ──→ بهبود تصویر تطبیقی (غبار / نور مستقیم / شب / تاری)
    ↓  PlateRecognizer (CRNN + CTC)  ← سه نسخه از هر برش در یک batch
  متن خام
    ↓  normalize → repair → validate
    ↓  PlateVoter  ← رأی‌گیری روی چند فریم (کلید دقت بالا)
  پلاک تأییدشده
    ↓  VehicleDB (SQLite)
 نمایش مشخصات راننده روی مانیتور
```

---

## ۱. نصب

```bash
cd license-plate-detection
python -m venv .venv && source .venv/bin/activate
```

اول PyTorch را متناسب با نسخه CUDA خود نصب کنید:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

سپس بقیه:

```bash
pip install -r requirements.txt
```

### فونت فارسی (الزامی برای نمایش روی مانیتور)

`cv2.putText` اصلاً فارسی نمی‌کشد. بدون فونت، نام راننده به‌صورت `??????` نمایش داده می‌شود.

```bash
sudo apt install fonts-vazir       # یا هر فونت فارسی دیگر
```

اگر فونت در مسیر غیرمعمول است، در `configs/config.yaml` مسیر آن را بدهید:

```yaml
display:
  font_path: "/usr/share/fonts/truetype/vazir/Vazir.ttf"
```

بررسی سلامت نصب:

```bash
python -c "import torch, cv2, ultralytics, arabic_reshaper; print('CUDA:', torch.cuda.is_available())"
```

---

## ۲. دیتاست — این مهم‌ترین بخش است

### ساختار مورد انتظار

دیتاست شما [IR-LPR](https://github.com/mut-deep/IR-LPR) است و باید دقیقاً این چیدمان را داشته باشد:

```
/mnt/g/Bistun-kavir/
├── train/            19381 عکس + XML   ← برش پلاک   → OCR
├── validation/        2805
├── test/              5559                          مجموع: 27,745
├── car_train/        14670 عکس + XML   ← صحنه کامل  → دتکتور
├── car_validation/    2120
└── car_test/          4176                          مجموع: 20,966
```

هر عکس یک `.xml` هم‌نام کنارش دارد.

| پوشه | محتوا | کدام مدل |
|---|---|---|
| `train` / `validation` / `test` | برش پلاک، بدون پسوند (`00005.jpg`) | **recognizer** (خواندن متن) |
| `car_*` | خودرو کامل، `day_00010.jpg` و `night (726).jpg` | **detector** (پیدا کردن پلاک) |

⚠️ توجه: تصاویر **خودرو** برای **دتکتور** هستند، نه recognizer. اگر جابه‌جا بدهید، دتکتور یاد می‌گیرد «کل تصویر پلاک است» و روی فریم دوربین هیچ پلاکی پیدا نمی‌کند.

اندازه‌گیری روی داده واقعی شما:

- برش پلاک: پلاک ۶۸٪ مساحت تصویر — برای OCR عالی، برای دتکشن بی‌فایده
- تصاویر خودرو: تصویر میانه ۹۶۰×۹۶۰ (روز) و ۱۲۸۰×۹۶۰ (شب)، پلاک **۰.۳ تا ۰.۸٪** مساحت، عرض میانه ۱۲۰ تا ۱۴۰ پیکسل — دقیقاً چیزی که دتکتور لازم دارد
- XMLهای `car_*` یک شیء صریح **`کل ناحیه پلاک`** دارند که کد از آن استفاده می‌کند
- حدود **۱۴٪** تصاویر خودرو بیش از یک پلاک دارند (تا ۶ تا)؛ هر کدام یک خط جدا در label می‌شود

### درباره سؤال شما: «باید اسم عکس را پلاک بگذاریم؟»

تا حدی درست است، ولی **کار دستی لازم ندارید**:

- برای **OCR**، بله — این پروژه برچسب را از نام فایل می‌خواند (`12ب34567.jpg`). این یک قرارداد همین پروژه است، نه یک قانون کلی.
- ولی متن پلاک همین حالا داخل XMLهای شماست. `prepare_dataset.py` آن را می‌خواند و **خودکار** نام فایل را می‌سازد.
- برای **دتکتور**، نام فایل اصلاً مهم نیست؛ فقط کادر پلاک مهم است.

پس شما هیچ فایلی را دستی تغییر نام نمی‌دهید.

### ساخت دیتاست‌ها — فقط یک دستور

با چیدمان بالا، همه مسیرها خودکار پیدا می‌شوند:

```bash
cd ~/bistun_kavir/license-plate-detection
python -m training.prepare_dataset --root /mnt/g/Bistun-kavir
```

اگر پوشه‌ها جای دیگری هستند، می‌توانید تک‌تک بدهید:

```bash
python -m training.prepare_dataset \
  --plate-train /path/train --plate-val /path/validation --plate-test /path/test \
  --car-train   /path/car_train --car-val /path/car_validation --car-test /path/car_test
```

خروجی:

```
data/ocr_dataset/       ← تصاویر با نام پلاک واقعی (برای recognizer)
data/plate_dataset/     ← فرمت YOLO: images/ + labels/ + data.yaml (برای دتکتور)
```

**این سه خط را حتماً چک کنید:**

```
Boxes from explicit plate objects: 20966     ← باید تقریباً همه باشد
Boxes from character union       : 0
Median plate coverage of image area: 0.6%    ← باید زیر ۵٪ باشد
```

اگر `coverage` بالای ۳۵٪ شد، هشدار چاپ می‌شود — یعنی اشتباهی برش پلاک را به‌عنوان منبع دتکشن داده‌اید.

---

## ۳. آموزش

### دتکتور

```bash
python -m training.train_detector
```

اگر روی برش پلاک اجرا شود، این هشدار را می‌بینید — جدی بگیرید:

```
*** WARNING ***
The plate fills most of each image, so this source is a set of plate CROPS...
```

### مدل خواندن پلاک

```bash
python -m training.train_recognizer
```

در ابتدای آموزش این خط را چاپ می‌کند:

```
CTC timesteps: 32 (longest label: 9)
```

اگر تعداد timestep از بلندترین برچسب کمتر باشد، برنامه **فوراً خطا می‌دهد** — چون CTC در آن حالت از نظر ریاضی نمی‌تواند یاد بگیرد. (این دقیقاً باگی بود که در نسخه قبلی وجود داشت: ۸ timestep برای ۹ کاراکتر.)

هر epoch گزارش می‌دهد و **بهترین** مدل را بر اساس CER ذخیره می‌کند:

```
Epoch 007 | loss 0.3812 | val CER 0.0241 | plate acc 0.9106  <- saved
```

- **CER** = نرخ خطای کاراکتر (کمتر بهتر) — هدف زیر ۰.۰۲
- **plate acc** = درصد پلاک‌هایی که کاملاً درست خوانده شده‌اند — هدف بالای ۰.۹۵

برای شرایط سخت معدن، augmentation واقعی در آموزش اعمال می‌شود: غبار اخرایی، فلاش خورشید، شب پرنویز، motion blur، پرسپکتیو و فشرده‌سازی JPEG.

خروجی هر دو مرحله در `weights/` ذخیره می‌شود.

---

## ۴. اجرا

```bash
python main.py                              # با تنظیمات configs/config.yaml
python main.py --source 0                   # وبکم
python main.py --source video.mp4           # فایل ویدیو (برای تست)
python main.py --source rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
```

کلید `q` یا `Esc` برای خروج.

اگر وزن‌ها موجود نباشند، پیام واضح می‌دهد و مراحل آموزش را یادآوری می‌کند.

---

## ۵. ثبت رانندگان

بدون این مرحله دیتابیس خالی است و مانیتور چیزی برای نمایش ندارد.

```bash
python -m tools.register_vehicle add \
  --plate "12ب34567" \
  --driver "علی محمدی" \
  --national-id 0012345678 \
  --truck-id TRK-001 \
  --model "بنز ۱۹۲۳" \
  --company "معدن سنگان"
```

ممنوع کردن یک خودرو:

```bash
python -m tools.register_vehicle add --plate "56د12345" --driver "..." --denied --note "مدارک ناقص"
```

ورود گروهی از CSV (ستون اجباری فقط `plate`):

```csv
plate,driver_name,national_id,truck_id,vehicle_model,company,allowed,note
12ب34567,علی محمدی,0012345678,TRK-001,بنز ۱۹۲۳,معدن سنگان,1,
34ج67890,رضا کریمی,0098765432,TRK-014,ولوو FH,معدن سنگان,1,
```

```bash
python -m tools.register_vehicle import drivers.csv
python -m tools.register_vehicle list
python -m tools.register_vehicle export backup.csv
```

پلاک‌های نامعتبر هنگام ورود رد می‌شوند و گزارش داده می‌شوند.

---

## ۵.۵. ثبت تصویر برای بازبینی (capture)

برای اینکه بشود تشخیص مدل را با خودروی واقعی مقایسه کرد، هر خودرویی که پلاکش **تأیید** می‌شود آرشیو تصویری می‌گیرد.

```yaml
capture:
  enabled: true
  output_dir: captures
  save_full_frame: true      # عکس خودرو — همان چیزی که مقایسه می‌کنید
  save_raw: true             # برش پلاک، همان‌طور که دتکتور بریده
  save_enhanced: true        # نسخه‌ای که OCR واقعاً خوانده
  annotate: true             # کشیدن کادر پلاک + اطمینان روی عکس کامل
  min_interval_seconds: 3.0
  max_files: 20000
  max_age_days: 30
```

خروجی:

```
captures/
├── 2026-09-02/
│   ├── 142421_324_12ب34567_full.jpg   ← عکس کل خودرو با کادر پلاک
│   ├── 142421_324_12ب34567_raw.jpg    ← برش پلاک
│   └── 142421_324_12ب34567_enh.jpg    ← نسخه بهبودیافته‌ای که خوانده شد
└── captures.csv
```

`captures.csv` برای بازبینی سریع در اکسل:

```csv
timestamp,plate,det_conf,ocr_conf,full_frame,plate_raw,plate_enhanced
2026-09-02 14:24:21,12ب34567,0.8800,0.9100,2026-09-02/142421_324_...full.jpg,...
```

می‌توانید بر اساس `ocr_conf` مرتب کنید و از کم‌اطمینان‌ترین‌ها شروع به بررسی کنید — همان‌جایی که خطاها جمع‌اند.

**سه نکته درباره رفتار آن:**

- **هر عبور خودرو = یک عکس.** تا وقتی کامیون جلوی دوربین ایستاده دوباره ذخیره نمی‌شود؛ اگر بعداً برگردد عکس جدید می‌گیرد. `min_interval_seconds` همین فاصله است.
- **نوشتن روی دیسک در ترد جداست.** کدگذاری سه JPEG ده‌ها میلی‌ثانیه طول می‌کشد؛ اگر داخل حلقه تشخیص انجام می‌شد به‌صورت فریم‌های افتاده دیده می‌شد. اگر دیسک کند شود، عکس دور ریخته می‌شود نه اینکه دروازه کند شود.
- **`max_files` و `max_age_days` را حتماً ست کنید.** یک دروازه ۲۴ ساعته وگرنه دیسک را پر می‌کند.

---

## ۶. راه‌اندازی روی دوربین معدن

### الف) گرفتن آدرس RTSP

از سازنده دوربین بگیرید. نمونه‌های رایج:

| برند | الگو |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |

تست مستقل قبل از هر چیز:

```bash
ffplay "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

### ب) نصب فیزیکی دوربین

این بخش بیشترین تأثیر را روی دقت دارد — بیشتر از هر تنظیم نرم‌افزاری:

- **ارتفاع و زاویه:** پلاک باید تا حد امکان روبه‌رو دیده شود. زاویه افقی زیر ۳۰ درجه.
- **اندازه پلاک در فریم:** عرض پلاک حداقل **۱۰۰ پیکسل**. اگر کمتر است، دوربین را نزدیک‌تر کنید یا zoom بدهید.
- **سرعت شاتر:** برای کامیون در حال حرکت، شاتر سریع (۱/۵۰۰ یا سریع‌تر) لازم است وگرنه پلاک تار می‌شود. این را در تنظیمات خود دوربین ست کنید.
- **نور شب:** IR illuminator روشن باشد. پلاک ایرانی بازتابنده است و در IR خوب دیده می‌شود.
- **خورشید مستقیم:** دوربین را طوری بگذارید که خورشید پشت آن نباشد. WDR دوربین را فعال کنید.

### ج) تنظیم config

```yaml
camera:
  source: "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
  width: 1280
  height: 720

database:
  seed_demo: false        # در محیط واقعی حتماً false

display:
  fullscreen: true
  hold_seconds: 5.0       # چند ثانیه مشخصات راننده روی صفحه بماند
```

### د) تنظیم دقت (بعد از دیدن نتیجه واقعی)

```yaml
detector:
  conf_threshold: 0.35    # پلاک را دیر پیدا می‌کند؟ کم کنید. تشخیص اشتباه زیاد؟ زیاد کنید.

preprocessing:
  min_votes: 3            # کندتر ولی مطمئن‌تر؟ زیاد کنید (مثلاً 5)
  vote_window_seconds: 2.0
  auto_enhance: true      # بهبود تطبیقی برای غبار/نور/شب
```

منطق رأی‌گیری: یک فریم ممکن است به خاطر غبار یا بازتاب نور خراب باشد، ولی این خطاها بین فریم‌ها همبسته نیستند در حالی که پلاک واقعی هست. سامانه تا وقتی چند فریم روی یک پلاک توافق نکنند، مشخصات راننده را نشان نمی‌دهد.

### ه) اجرای خودکار هنگام بوت (systemd)

```bash
sudo nano /etc/systemd/system/lpr-gate.service
```

```ini
[Unit]
Description=Mine Gate License Plate Recognition
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lucifer666
WorkingDirectory=/home/lucifer666/bistun_kavir/license-plate-detection
Environment="DISPLAY=:0"
ExecStart=/home/lucifer666/bistun_kavir/license-plate-detection/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=graphical.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lpr-gate
sudo systemctl status lpr-gate
journalctl -u lpr-gate -f          # مشاهده لاگ زنده
```

قطع شبکه دوربین باعث توقف سرویس نمی‌شود — `FrameGrabber` خودش تلاش مجدد می‌کند و روی صفحه «قطع ارتباط دوربین» نشان می‌دهد.

---

## ۷. مسیر رسیدن به محصول نهایی

- [ ] نصب وابستگی‌ها + فونت فارسی
- [ ] `python -m training.prepare_dataset --root /mnt/g/Bistun-kavir`
- [ ] آموزش دتکتور — بررسی mAP در `runs/detect/iran_plate/`
- [ ] آموزش OCR — رسیدن به CER زیر ۰.۰۲
- [ ] تست با یک ویدیوی ضبط‌شده از خود معدن (`--source video.mp4`)
- [ ] نصب فیزیکی دوربین طبق راهنمای بالا
- [ ] ثبت رانندگان واقعی + `seed_demo: false`
- [ ] تنظیم `conf_threshold` و `min_votes` بر اساس نتیجه واقعی
- [ ] راه‌اندازی systemd
- [ ] **مرحله طلایی:** جمع‌آوری ۵۰۰ تا ۱۰۰۰ تصویر از دوربین *خود معدن* (شب، غبار، آفتاب) و fine-tune کردن هر دو مدل روی آن‌ها

آخرین مورد بیشترین تأثیر را دارد. هیچ دیتاست عمومی شرایط نوری و گرد و غبار معدن شما را ندارد.

---

## ۸. عیب‌یابی

| نشانه | علت / راه‌حل |
|---|---|
| نام راننده `??????` است | فونت فارسی نصب نیست → `display.font_path` را ست کنید |
| حروف فارسی جدا جدا | `pip install arabic-reshaper python-bidi` |
| هیچ پلاکی پیدا نمی‌شود | دتکتور روی برش پلاک آموزش دیده → زیرمجموعه car لازم است |
| پلاک پیدا می‌شود ولی متن غلط | CER مدل بالاست → آموزش بیشتر / fine-tune با داده معدن |
| پلاک پیدا می‌شود ولی مشخصات نمی‌آید | پلاک در دیتابیس ثبت نشده → `register_vehicle list` |
| مشخصات چشمک می‌زند | `hold_seconds` را زیاد کنید |
| تأخیر روی RTSP | `detector.img_size` را به ۴۸۰ کم کنید یا `half: true` |
| `CUDA out of memory` | `training.recognizer.batch` را کم کنید (۳۲ → ۱۶) |
| خطای CTC timesteps | `recognizer.img_width` را زیاد کنید |
| تصویر کند / پرش دارد | `detector.max_det` را کم کنید؛ GPU را بررسی کنید |

---

## ۹. ساختار پروژه

```
license-plate-detection/
├── main.py                      نقطه ورود
├── requirements.txt
├── configs/config.yaml          همه تنظیمات
├── models/
│   ├── detector.py              YOLOv8 + padding کادر + fp16
│   ├── recognizer.py            ResNet-CRNN + رمزگشایی CTC با اطمینان
│   └── pipeline.py              چند-نسخه‌ای + PlateVoter
├── utils/
│   ├── image_processing.py      بهبود تطبیقی (غبار/نور/شب/تاری)
│   ├── plate_utils.py           نرمال‌سازی، اعتبارسنجی، ترمیم
│   ├── database.py              رجیستری SQLite
│   └── overlay.py               رندر متن فارسی
│   └── plate_saver.py           آرشیو تصویری برای بازبینی (ترد جدا)
├── inference/realtime.py        FrameGrabber + حلقه نمایش
├── training/
│   ├── prepare_dataset.py       ساخت هر دو دیتاست از XML
│   ├── train_detector.py
│   └── train_recognizer.py      + augmentation شرایط سخت + CER
└── tools/register_vehicle.py    مدیریت رانندگان
```

---

## ۱۰. تغییرات این بازبینی

### باگ‌های بحرانی (سامانه اصلاً کار نمی‌کرد)

1. **CTC از نظر ریاضی غیرممکن بود** — ResNet با stride ۳۲، عرض ۲۵۶ را به ۸ timestep کاهش می‌داد، ولی پلاک ایرانی ۸ تا ۹ کاراکتر دارد. آموزش هرگز همگرا نمی‌شد. stride عرض در layer3/layer4 آزاد شد → ۳۲ timestep، به‌علاوه یک چک که در صورت تکرار، فوراً خطا بدهد.

2. **هر پلاک ۹۰ درجه می‌چرخید** — در `correct_perspective` ترتیب نقاط مقصد با خروجی `_order_points` نمی‌خواند. ضمناً هر کانتوری را قبول می‌کرد (معمولاً یک حرف، نه قاب پلاک). حالا اعتبارسنجی نسبت ابعاد و پوشش اضافه شده و در غیر این صورت به deskew ساده برمی‌گردد.

3. **ناسازگاری آموزش و استنتاج** — آموزش با RGB رنگی، ولی `enhance_plate` خروجی خاکستری می‌داد و `_preprocess` آن را BGR تفسیر می‌کرد.

4. **کادرهای YOLO خارج از تصویر** — `write_yolo_label` مقادیر نرمال‌شده را جدا جدا clamp می‌کرد؛ کادری با مرکز ۰.۴۷ و عرض ۱.۰ مستطیل دیگری توصیف می‌کند. روی داده واقعی: ۲۴۹ از ۳۰۰ کادر خارج از محدوده بود → حالا ۰ از ۳۰۰.

### اصلاح برچسب‌ها (تأییدشده روی هر ۲۷٬۷۴۵ فایل شما)

| مشکل | تعداد | نتیجه |
|---|---|---|
| `_2.jpg` تکراری‌ها بی‌صدا حذف می‌شدند | ۲٬۲۵۵ | بازیابی شد |
| `'الف'` به‌جای `ا` سه حرف می‌ماند | ۷۸ | اصلاح شد |
| `'ژ (معلولین و جانبازان)'` با فاصله و پرانتز | ۳۴ | اصلاح شد |
| `'ه‍'` با zero-width joiner | ۱٬۶۳۲ | اصلاح شد |

نمونه‌های قابل استفاده: **۲۳٬۵۶۷ → ۲۵٬۸۲۴**

### پایداری و عملکرد

- **نشت اتصال SQLite** — هر فریم یک اتصال باز می‌کرد و هرگز نمی‌بست (`with` فقط commit می‌کند). حالا اتصال دائمی + کش.
- **`seed_demo` رکورد واقعی سه راننده را در هر بوت بازنویسی می‌کرد** → `INSERT OR IGNORE` و پیش‌فرض خاموش.
- **یک فریم ناموفق دوربین کل برنامه را می‌کشت** → ترد جدا با اتصال مجدد خودکار و حذف فریم‌های کهنه RTSP.
- **`fliplr` پیش‌فرض اولتراlytics ۰.۵ است** — نصف مواقع پلاک آینه‌ای یاد می‌گرفت. صفر شد.
- **`best.replace()`** بین فایل‌سیستم‌ها fail می‌کند → `copy2`.
- **آموزش OCR نه validation داشت نه best-checkpoint** (فقط epoch آخر) → split + CER + accuracy + ذخیره بهترین + AMP.
- **کل بخش `preprocessing` در config خوانده می‌شد ولی هرگز استفاده نمی‌شد** → حالا واقعاً وصل است.
- fp16، warmup، و انتقال یکجای تنسور به‌جای `.item()` به ازای هر کادر.

### قابلیت‌های جدید

- **رأی‌گیری چندفریمی** (`PlateVoter`) — کلید دقت بالا در شرایط سخت.
- **OCR چند-نسخه‌ای** — سه رندر از هر برش در یک batch؛ نسخه‌ای که ساختار پلاک معتبر دارد برنده می‌شود، حتی اگر اطمینان عددی کمتری داشته باشد.
- **بهبود تصویر تطبیقی** — inpaint بازتاب خورشید، تخت‌کردن گرادیان نور، gamma شب، unsharp متناسب با تاری.
- **رندر متن فارسی** — قبلاً نام راننده روی مانیتور `??????` بود.
- **`repair_plate`** — بازسازی پلاک از خوانش کمی خراب CTC.
- **`tools/register_vehicle.py`** — مدیریت رانندگان (add/list/import/export).

---

## نکته مهم درباره وزن‌های قبلی

معماری recognizer تغییر کرده، پس **باید دوباره آموزش دهید**. پوشه `weights/` اصلاً وجود نداشت، بنابراین چیزی از دست نمی‌رود.
