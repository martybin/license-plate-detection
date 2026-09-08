# راهنمای گام‌به‌گام اجرا — از صفر تا دوربین معدن

این فایل فقط دستورهاست، به ترتیبی که باید اجرا شوند. توضیح مفصل هر بخش در [README.md](README.md) است.

**همه‌ی دستورها از ریشه‌ی پروژه اجرا می‌شوند:**

```bash
cd ~/bistun_kavir/license-plate-detection
```

⚠️ **روی Linux یا WSL اجرا کنید، نه ویندوز نیتیو.** کتابخانه‌ی ultralytics روی ویندوز موقع import خطا می‌دهد (`OSError: /etc/os-release`) — باگ خودشان است، نه این پروژه.

---

## گام ۰ — پیش‌نیازها

| نیاز | حداقل |
|---|---|
| GPU | کارت NVIDIA با ۶ گیگ VRAM یا بیشتر (مثلاً RTX A2000 12GB) |
| RAM | ۱۶ گیگ |
| فضای دیسک | ۵۰ گیگ آزاد |
| سیستم‌عامل | Ubuntu / WSL2 |

**دیتاست را روی دیسک محلی بگذارید، نه درایو شبکه.** خواندن از درایو شبکه آموزش را چند برابر کند می‌کند.

---

## گام ۱ — نصب (یک‌بار)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### اول نسخه‌ی CUDA درایور را ببینید

```bash
nvidia-smi
```

گوشه‌ی بالا-راست می‌نویسد `CUDA Version: 12.x`. این **حداکثر** نسخه‌ای است که درایور شما پشتیبانی می‌کند؛ build پایتورچ باید مساوی یا کمتر از آن باشد.

اگر `nvidia-smi` اصلاً اجرا نشد، درایور NVIDIA نصب نیست — اول آن را نصب کنید.

### بعد PyTorch نسخه‌ی GPU را نصب کنید

`cu121` یعنی build مخصوص CUDA 12.1. اگر درایور شما نسخه‌ی دیگری می‌خواهد، دستور دقیق را از
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) بگیرید.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

⚠️ نسخه‌ی `whl/cpu` را نصب نکنید — آموزش روی CPU به‌جای چند ساعت، چند روز طول می‌کشد.

بقیه‌ی وابستگی‌ها:

```bash
pip install -r requirements.txt
```

فونت فارسی (بدون این، نام راننده روی مانیتور `??????` می‌شود):

```bash
sudo apt update && sudo apt install -y fonts-vazir
```

### بررسی نصب

```bash
python -c "import torch, cv2, ultralytics, arabic_reshaper; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

خروجی درست:

```
2.5.1+cu121
CUDA: True
NVIDIA RTX A2000 12GB
```

**دو علامت خطر — در هر دو حالت آموزش را شروع نکنید:**

| نشانه | یعنی |
|---|---|
| نسخه به `+cpu` ختم شود یا `+cu` نداشته باشد | build مخصوص CPU نصب شده |
| `CUDA: False` | یا build غلط است یا درایور مشکل دارد |

---

## گام ۲ — ساخت دیتاست

ساختار پوشه‌ها باید این باشد:

```
/path/to/dataset/
├── train/  validation/  test/          ← برش پلاک (برای OCR)
└── car_train/  car_validation/  car_test/   ← صحنه‌ی کامل خودرو (برای دتکتور)
```

```bash
python -m training.prepare_dataset --root /path/to/dataset
```

### ✅ چه چیزی باید ببینید

```
Boxes from explicit plate objects: ~20966      ← باید تقریباً همه باشد
Boxes from character union       : 0
Median plate coverage of image area: ~0.6%     ← باید زیر ۵٪ باشد
cropped N image(s) whose region panel was corrupted
Total success: ~55000 | Total failed: ~3000
```

### ❌ اگر این را دیدید، متوقف شوید

```
*** WARNING ***
The plate fills most of each image, so this source is a set of plate CROPS...
```

یعنی پوشه‌های `car_*` اشتباه داده شده‌اند. مسیرها را چک کنید.

خروجی: `data/plate_dataset/` و `data/ocr_dataset/`

---

## گام ۳ — آموزش دتکتور (پیدا کردن پلاک)

```bash
python -m training.train_detector
```

⏱️ حدود **۵ تا ۷ ساعت** روی RTX A2000. خروجی: `weights/yolov8_plate.pt`

**چک کنید:** `mAP50` در انتهای خروجی باید بالای **۰.۹** باشد. نتایج کامل در `runs/detect/iran_plate/`.

اگر `CUDA out of memory` گرفتید، در `configs/config.yaml`:

```yaml
training:
  detector:
    batch: 8        # از ۱۶ کم کنید
```

---

## گام ۴ — آموزش خواننده‌ی پلاک

```bash
python -m training.train_recognizer
```

⏱️ حدود **۳ تا ۶ ساعت**. خروجی: `weights/resnet_crnn_iran.pt`

### ✅ خط اول باید این باشد

```
CTC timesteps: 24 (longest label: 9)
```

اگر عدد اول از عدد دوم کمتر بود، برنامه خودش خطا می‌دهد و متوقف می‌شود.

### هر epoch

```
Epoch 007 | loss 0.38 | train CER 0.019 | val CER 0.024 | plate acc 0.91 | ع acc 0.95  <- saved
```

- `val CER` هدف: **زیر ۰.۰۲**
- `plate acc` هدف: **بالای ۰.۹۵**

### در پایان — مهم‌ترین خط

```
Final train CER 0.0198 vs val CER 0.0241  (gap +0.0043) -> balanced
```

| اگر گفت | یعنی | چه کنید |
|---|---|---|
| `balanced` | خوب است | ادامه دهید |
| `underfitting` | حتی داده‌ی آموزشی را هم فیت نکرده | `epochs: 100` کنید و دوباره اجرا کنید |
| `overfitting` | روی train خوب، روی val بد | داده‌ی بیشتر لازم است → گام ۸ |

و جدول per-letter نشان می‌دهد کدام حرف ضعیف است.

---

## گام ۵ — تست مدل قبل از رفتن سر دوربین

روی یک ویدیو یا وبکم:

```bash
python -m tools.live_test                    # وبکم لپ‌تاپ
python -m tools.live_test --image plate.jpg  # یک عکس
python -m tools.live_test --ocr-only         # فقط OCR، بدون دتکتور
```

کلیدها: `q` خروج · `s` ذخیره‌ی عکس · `space` مکث · `r` صفر کردن رأی‌ها

---

## گام ۶ — ثبت رانندگان

بدون این، مانیتور چیزی برای نمایش ندارد.

```bash
python -m tools.register_vehicle add \
  --plate "12ع44964" \
  --driver "علی محمدی" \
  --national-id 0012345678 \
  --truck-id TRK-001 \
  --model "بنز ۱۹۲۳" \
  --company "معدن سنگان"
```

خودروی ممنوع:

```bash
python -m tools.register_vehicle add --plate "56د12345" --driver "..." --denied --note "مدارک ناقص"
```

ورود گروهی از فایل CSV (فقط ستون `plate` اجباری است):

```csv
plate,driver_name,national_id,truck_id,vehicle_model,company,allowed,note
12ع44964,علی محمدی,0012345678,TRK-001,بنز ۱۹۲۳,معدن سنگان,1,
```

```bash
python -m tools.register_vehicle import drivers.csv
python -m tools.register_vehicle list        # نمایش همه
python -m tools.register_vehicle export backup.csv
```

---

## گام ۷ — اتصال به دوربین

### الف) آدرس RTSP را بگیرید

| برند | الگو |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |

**اول مستقل از پروژه تست کنید:**

```bash
ffplay "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

اگر اینجا تصویر نیامد، مشکل از شبکه یا دوربین است نه از کد.

### ب) در `configs/config.yaml` بگذارید

```yaml
camera:
  source: "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
  width: 1280
  height: 720
  fps: 25

database:
  seed_demo: false          # در محیط واقعی حتماً false

display:
  fullscreen: true
  hold_seconds: 5.0
  font_path: ""             # اگر فونت پیدا نشد، مسیر .ttf را بدهید
```

### ج) نصب فیزیکی دوربین — بیشترین اثر روی دقت

- **زاویه:** پلاک تا حد امکان روبه‌رو. زاویه‌ی افقی زیر ۳۰ درجه.
- **اندازه:** عرض پلاک در تصویر حداقل **۱۰۰ پیکسل**.
- **شاتر:** برای کامیون در حرکت، **۱/۵۰۰ ثانیه یا سریع‌تر** — وگرنه پلاک تار می‌شود.
- **شب:** IR روشن باشد. پلاک ایرانی بازتابنده است و در IR خوب دیده می‌شود.
- **آفتاب:** خورشید پشت دوربین نباشد. WDR دوربین را فعال کنید.

---

## گام ۸ — افزودن عکس‌های واقعی معدن ⭐ (برای دقت معدن مهم‌ترین گام است)

### چرا این گام لازم است

پوشه‌ی `plates/` (عکس‌های خام دوربین معدن) **در گام ۲ خوانده نمی‌شود** و مدل از آن چیزی یاد نمی‌گیرد. دو دلیل دارد:

1. `prepare_dataset` فقط `train/`, `validation/`, `test/` و `car_*` را می‌خواند — `plates/` در فهرستش نیست.
2. آن عکس‌ها **هیچ فایل XML ندارند**. اسمشان timestamp دوربین است، نه متن پلاک. یعنی برچسب ندارند.

این گام همان چیزی است که آن‌ها را به داده‌ی آموزشی تبدیل می‌کند.

### چرا برای معدن حیاتی است

در دیتاست IR-LPR حرف `ع` فقط حدود **۲.۲٪** نمونه‌ها را می‌گیرد، در حالی که ناوگان معدن شما تقریباً همه‌اش پلاک زرد عمومی با `ع` است. بعد از برچسب زدن این ~۶۰۰۰ عکس، سهم `ع` به حدود **۲۰٪** می‌رسد و از کمیاب‌ترین حرف به پرتکرارترین تبدیل می‌شود.

ضمناً این عکس‌ها از **همان دوربین و همان نور و همان غبار** معدن هستند — چیزی که هیچ دیتاست عمومی ندارد.

```yaml
# configs/config.yaml → موقتاً:
recognizer:
  allowed_letters: "ع"
```

```bash
# اول با نمونه‌ی کوچک ببینید نرخ پذیرش چقدر است
python -m tools.autolabel_plates label --source /path/to/plates --limit 200 --dry-run

# اجرای واقعی
python -m tools.autolabel_plates label --source /path/to/plates

# ستون corrected_plate را در فایل زیر پر کنید، بعد:
python -m tools.autolabel_plates import review/needs_review.csv
```

```yaml
# ⚠️ حتماً برگردانید:
  allowed_letters: ""
```

بعد دوباره `python -m training.train_recognizer` اجرا کنید.

---

## گام ۹ — اجرا

```bash
python main.py
```

یا با منبع دلخواه:

```bash
python main.py --source 0                       # وبکم
python main.py --source video.mp4               # فایل ویدیو
python main.py --source "rtsp://user:pass@IP:554/..."
```

خروج: `q` یا `Esc`

---

## گام ۱۰ — اجرای خودکار هنگام روشن شدن (systemd)

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
```

`Restart=always` یعنی اگر برنامه به هر دلیلی بسته شد، ۱۰ ثانیه بعد خودش بالا می‌آید.

---

## گام ۱۱ — مانیتورینگ

### لاگ زنده

```bash
journalctl -u lpr-gate -f              # لاگ لحظه‌ای
journalctl -u lpr-gate --since "1 hour ago"
journalctl -u lpr-gate -p err          # فقط خطاها
```

### روی خود مانیتور

پایین صفحه همیشه این نوار هست:

```
دوربین متصل   |   24.3 FPS
```

اگر بنویسد **«قطع ارتباط دوربین»** (قرمز)، برنامه زنده است ولی دوربین نمی‌دهد — خودش هر ۲ ثانیه تلاش مجدد می‌کند.

### عکس‌های ثبت‌شده برای بازبینی

```
captures/
├── 2026-09-08/
│   ├── 142421_324_12ع44964_full.jpg   ← عکس کل خودرو
│   ├── 142421_324_12ع44964_raw.jpg    ← برش پلاک
│   └── 142421_324_12ع44964_enh.jpg    ← نسخه‌ای که OCR خواند
└── captures.csv
```

**برای بررسی دقت، `captures.csv` را در اکسل باز کنید و بر اساس `ocr_conf` صعودی مرتب کنید** — کم‌اطمینان‌ترین‌ها بالا می‌آیند و خطاها همان‌جا جمع‌اند.

آخرین تشخیص‌ها از خط فرمان:

```bash
tail -5 captures/captures.csv
ls -lt captures/$(date +%F)/ | head
```

### سلامت سیستم

```bash
systemctl is-active lpr-gate           # باید active بگوید
nvidia-smi                             # مصرف GPU
df -h .                                # فضای دیسک
du -sh captures/                       # حجم عکس‌های ذخیره‌شده
```

⚠️ **حتماً محدودیت فضا را در config بگذارید** وگرنه دیسک پر می‌شود:

```yaml
capture:
  max_files: 20000
  max_age_days: 30
```

---

## گام ۱۲ — تنظیم دقت بعد از دیدن نتیجه‌ی واقعی

```yaml
detector:
  conf_threshold: 0.35    # پلاک را دیر پیدا می‌کند؟ کم کنید (۰.۲۵)
                          # تشخیص اشتباه زیاد؟ زیاد کنید (۰.۵)

preprocessing:
  min_votes: 3            # کندتر ولی مطمئن‌تر؟ زیاد کنید (۵)
  vote_window_seconds: 2.0
```

بعد از هر تغییر:

```bash
sudo systemctl restart lpr-gate
```

---

## عیب‌یابی سریع

| نشانه | علت / راه‌حل |
|---|---|
| `OSError: /etc/os-release` | ویندوز نیتیو → روی Linux/WSL اجرا کنید |
| `CUDA: False` | درایور یا نسخه‌ی CUDA اشتباه است |
| نام راننده `??????` | فونت فارسی نصب نیست → `apt install fonts-vazir` |
| حروف فارسی جدا جدا | `pip install arabic-reshaper python-bidi` |
| هیچ پلاکی پیدا نمی‌شود | `conf_threshold` را کم کنید؛ اندازه‌ی پلاک در فریم را چک کنید |
| پلاک پیدا می‌شود، متن غلط | مدل بیشتر آموزش لازم دارد → گام ۴ و ۸ |
| پلاک درست، ولی مشخصات نمی‌آید | در دیتابیس ثبت نشده → `register_vehicle list` |
| مشخصات چشمک می‌زند | `hold_seconds` را زیاد کنید |
| تأخیر روی RTSP | `detector.img_size` را ۴۸۰ کنید |
| `CUDA out of memory` | `batch` را نصف کنید |
| دیسک پر شد | `max_files` / `max_age_days` را ست کنید |

---

## چک‌لیست نهایی قبل از تحویل

- [ ] `CUDA: True`
- [ ] `prepare_dataset` بدون هشدار تمام شد
- [ ] دتکتور: `mAP50 > 0.9`
- [ ] خواننده: `val CER < 0.02` و `plate acc > 0.95`
- [ ] `Final train CER ... -> balanced`
- [ ] `allowed_letters: ""` (خالی)
- [ ] `seed_demo: false`
- [ ] رانندگان واقعی ثبت شده‌اند
- [ ] `max_files` و `max_age_days` ست شده
- [ ] گام ۸ انجام شده (عکس‌های `plates/` برچسب خورده و مدل دوباره آموزش دیده)
- [ ] با ویدیوی واقعی معدن تست شده
- [ ] سرویس systemd فعال و `Restart=always`
- [ ] بعد از ریست دستگاه، خودکار بالا می‌آید

---

## تست‌ها

هر وقت کد را تغییر دادید:

```bash
pip install pytest
python -m pytest
```

باید `291 passed` بدهد.
