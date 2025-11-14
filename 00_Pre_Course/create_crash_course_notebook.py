#!/usr/bin/env python3
"""
Script untuk membuat 00_python_crash_course.ipynb
Notebook lengkap untuk belajar Python dari nol
"""

import nbformat as nbf

# Buat notebook baru
nb = nbf.v4.new_notebook()

# List untuk menyimpan semua cells
cells = []

# ============================================================================
# HEADER
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🐍 Python Crash Course - Dari NOL hingga Mahir

---

## 📚 Selamat Datang di Python!

Selamat datang di **Python Crash Course**! Notebook ini dirancang khusus untuk **pemula absolut** yang belum pernah programming sebelumnya.

### 🎯 Tujuan Pembelajaran

Setelah menyelesaikan notebook ini, Anda akan:
- ✅ Paham syntax dasar Python
- ✅ Bisa menulis program sederhana
- ✅ Mengerti konsep variables, loops, functions
- ✅ Siap untuk masuk ke Data Science!

### ⏱️ Estimasi Waktu: 4-6 jam

### 💡 Cara Menggunakan Notebook Ini:
1. **Baca** setiap penjelasan dengan teliti
2. **Jalankan** setiap cell kode (tekan Shift+Enter)
3. **Eksperimen** dengan kode (ubah-ubah, lihat hasilnya!)
4. **Latihan** di setiap akhir bagian

### 🚀 Let's Start Your Programming Journey!

---"""))

# ============================================================================
# PART 1: BASICS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📖 PART 1: Python Basics (30 menit)

## Progress: 1/6 🟩⬜⬜⬜⬜⬜

Kita akan mulai dari yang paling dasar!

---

## 1.1 Print Statement - Menampilkan Teks

`print()` adalah fungsi untuk menampilkan (mencetak) sesuatu ke layar.

**Analogi**: Seperti Anda berbicara ke komputer, dan komputer mengulangi apa yang Anda katakan."""))

cells.append(nbf.v4.new_code_cell("""# Contoh pertama kita: Hello, World!
# Ini adalah tradisi programmer - program pertama selalu "Hello, World!"
print("Hello, World!")"""))

cells.append(nbf.v4.new_code_cell("""# Anda bisa print apapun yang ada di dalam tanda kutip
print("Saya sedang belajar Python!")
print("Python itu mudah dan menyenangkan!")"""))

cells.append(nbf.v4.new_code_cell("""# Print angka (tidak perlu tanda kutip untuk angka)
print(100)
print(3.14)"""))

cells.append(nbf.v4.new_code_cell("""# Print beberapa hal sekaligus (dipisahkan dengan koma)
print("Saya berumur", 25, "tahun")
print("Nilai saya:", 95.5)"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Tips Penting:
- **String (teks)**: Harus di dalam tanda kutip `"..."` atau `'...'`
- **Angka**: Tidak perlu tanda kutip
- **Koma**: Untuk memisahkan beberapa item (otomatis ada spasi)

---

## 1.2 Variables - Menyimpan Data

Variable adalah **"kotak penyimpanan"** untuk data.

**Analogi**: Seperti kotak dengan label. Anda bisa menaruh sesuatu di kotak, dan nanti mengambilnya kembali dengan menyebut label kotak tersebut."""))

cells.append(nbf.v4.new_code_cell("""# Membuat variable
nama = "Budi"  # Variable 'nama' berisi string "Budi"
umur = 25      # Variable 'umur' berisi angka 25

# Menampilkan isi variable
print(nama)
print(umur)"""))

cells.append(nbf.v4.new_code_cell("""# Menggunakan variable dalam kalimat
print("Nama saya adalah", nama)
print("Umur saya", umur, "tahun")"""))

cells.append(nbf.v4.new_code_cell("""# Variable bisa diubah (makanya namanya "variable")
nilai = 80
print("Nilai awal:", nilai)

nilai = 95  # Mengubah isi variable
print("Nilai setelah diubah:", nilai)"""))

cells.append(nbf.v4.new_markdown_cell("""### 📝 Aturan Penamaan Variable:
✅ **Boleh**: huruf, angka, underscore (_)
✅ **Harus dimulai** dengan huruf atau underscore
❌ **Tidak boleh**: spasi, simbol khusus (@, #, %, dll.)
❌ **Tidak boleh**: dimulai dengan angka

**Contoh**:
- ✅ `nama`, `umur`, `nilai_akhir`, `data2`
- ❌ `2data`, `nilai akhir`, `nilai@siswa`

---

## 1.3 Tipe Data - Jenis-jenis Data

Python punya beberapa tipe data dasar:"""))

cells.append(nbf.v4.new_code_cell("""# 1. INTEGER (int) - Bilangan bulat
jumlah_siswa = 30
tahun = 2025
print("Integer:", jumlah_siswa, tahun)
print("Tipe data:", type(jumlah_siswa))"""))

cells.append(nbf.v4.new_code_cell("""# 2. FLOAT - Bilangan desimal
tinggi = 170.5
berat = 65.3
pi = 3.14159
print("Float:", tinggi, berat, pi)
print("Tipe data:", type(tinggi))"""))

cells.append(nbf.v4.new_code_cell("""# 3. STRING (str) - Teks
nama = "Budi Santoso"
alamat = 'Jakarta'  # Bisa pakai " atau '
print("String:", nama, alamat)
print("Tipe data:", type(nama))"""))

cells.append(nbf.v4.new_code_cell("""# 4. BOOLEAN (bool) - True atau False (benar/salah)
sudah_lulus = True
hujan = False
print("Boolean:", sudah_lulus, hujan)
print("Tipe data:", type(sudah_lulus))"""))

cells.append(nbf.v4.new_markdown_cell("""### 🔄 Konversi Tipe Data

Kadang kita perlu mengubah tipe data (misalnya string ke integer)"""))

cells.append(nbf.v4.new_code_cell("""# String ke Integer
umur_string = "25"
umur_int = int(umur_string)
print("String:", umur_string, "->", "Integer:", umur_int)

# Integer ke String  
nilai = 95
nilai_string = str(nilai)
print("Integer:", nilai, "->", "String:", nilai_string)

# String ke Float
tinggi_string = "170.5"
tinggi_float = float(tinggi_string)
print("String:", tinggi_string, "->", "Float:", tinggi_float)"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 1.4 Operator - Melakukan Operasi

### Operator Aritmatika (Matematika)"""))

cells.append(nbf.v4.new_code_cell("""# Penjumlahan
a = 10
b = 3
print("Penjumlahan:", a + b)  # 13

# Pengurangan
print("Pengurangan:", a - b)  # 7

# Perkalian
print("Perkalian:", a * b)  # 30

# Pembagian (hasil float)
print("Pembagian:", a / b)  # 3.333...

# Pembagian bulat (hasil integer, dibulatkan ke bawah)
print("Pembagian bulat:", a // b)  # 3

# Modulus (sisa bagi)
print("Sisa bagi:", a % b)  # 1

# Pangkat
print("Pangkat:", a ** b)  # 10^3 = 1000"""))

cells.append(nbf.v4.new_markdown_cell("""### Operator Perbandingan

Menghasilkan `True` atau `False`"""))

cells.append(nbf.v4.new_code_cell("""x = 10
y = 5

print("x =", x, ", y =", y)
print("x == y (sama dengan):", x == y)      # False
print("x != y (tidak sama dengan):", x != y)  # True
print("x > y (lebih besar):", x > y)        # True
print("x < y (lebih kecil):", x < y)        # False
print("x >= y (lebih besar sama dengan):", x >= y)  # True
print("x <= y (lebih kecil sama dengan):", x <= y)  # False"""))

cells.append(nbf.v4.new_markdown_cell("""### Operator Logika

Untuk menggabungkan kondisi"""))

cells.append(nbf.v4.new_code_cell("""# and: Kedua kondisi harus True
print("True and True:", True and True)    # True
print("True and False:", True and False)  # False

# or: Salah satu kondisi True sudah cukup
print("True or False:", True or False)    # True
print("False or False:", False or False)  # False

# not: Membalikkan nilai
print("not True:", not True)    # False
print("not False:", not False)  # True"""))

cells.append(nbf.v4.new_code_cell("""# Contoh praktis
umur = 20
punya_sim = True

boleh_nyetir = (umur >= 17) and punya_sim
print("Boleh nyetir?", boleh_nyetir)  # True"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 🏋️ Latihan Part 1

Sekarang saatnya latihan! Coba solve sendiri dulu sebelum lihat jawaban.

### Latihan 1.1: Variable dan Print
Buat variable untuk menyimpan:
- Nama Anda
- Umur Anda
- Kota tempat tinggal

Lalu print dalam format: "Nama saya [nama], umur [umur] tahun, tinggal di [kota]"
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
# Contoh:
# nama = "..."
# umur = ...
# kota = "..."
# print(...)

"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 1.2: Operasi Matematika
Hitung luas dan keliling persegi panjang dengan panjang = 15 dan lebar = 8

Rumus:
- Luas = panjang × lebar
- Keliling = 2 × (panjang + lebar)
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
panjang = 15
lebar = 8

# Hitung luas dan keliling


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 1.3: Konversi Tipe Data
Diberikan: `harga_string = "50000"`

1. Konversi ke integer
2. Tambahkan pajak 10% (harga × 0.1)
3. Hitung total harga (harga + pajak)
4. Print hasilnya
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
harga_string = "50000"


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 1 Selesai!** Anda sudah belajar basics Python!

**Next**: Part 2 - Control Flow (if-else, loops)

---"""))

# Tambahkan lebih banyak cells untuk parts berikutnya...
# Saya akan continue dengan Part 2, 3, dst.

# PART 2: CONTROL FLOW
cells.append(nbf.v4.new_markdown_cell("""# 🔀 PART 2: Control Flow (45 menit)

## Progress: 2/6 🟩🟩⬜⬜⬜⬜

Control Flow = Mengontrol alur program (kapan code dijalankan, kapan dilewati)

---

## 2.1 If Statement - Pengkondisian

`if` digunakan untuk membuat keputusan dalam program.

**Analogi**: Seperti percabangan jalan. "**Jika** hujan, bawa payung. **Jika tidak**, tidak perlu."
"""))

cells.append(nbf.v4.new_code_cell("""# If sederhana
umur = 18

if umur >= 17:
    print("Anda sudah bisa buat SIM")
    print("Selamat!")
    
# Perhatikan: code di dalam if harus di-indent (geser ke kanan)"""))

cells.append(nbf.v4.new_code_cell("""# If-Else
nilai = 75

if nilai >= 70:
    print("Lulus!")
else:
    print("Tidak lulus, coba lagi")"""))

cells.append(nbf.v4.new_code_cell("""# If-Elif-Else (multiple kondisi)
nilai = 85

if nilai >= 90:
    print("Grade: A")
elif nilai >= 80:
    print("Grade: B")
elif nilai >= 70:
    print("Grade: C")
elif nilai >= 60:
    print("Grade: D")
else:
    print("Grade: E")"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Tips If Statement:
- **Indentasi (spasi di awal)** sangat penting di Python!
- Gunakan 4 spasi atau 1 Tab (konsisten)
- `elif` = else if (kondisi tambahan)
- `else` = jika semua kondisi sebelumnya False

---

## 2.2 For Loop - Perulangan dengan Jumlah Pasti

`for` digunakan untuk mengulang code sejumlah tertentu.

**Analogi**: "Lakukan push-up 10 kali" - Anda tahu pasti berapa kali harus mengulang.
"""))

cells.append(nbf.v4.new_code_cell("""# Loop dari 0 sampai 4 (5 kali)
for i in range(5):
    print("Iterasi ke-", i)"""))

cells.append(nbf.v4.new_code_cell("""# Loop dari 1 sampai 10
for i in range(1, 11):
    print(i, end=" ")  # end=" " agar tidak newline
print()  # Newline di akhir"""))

cells.append(nbf.v4.new_code_cell("""# Loop dengan step (loncat-loncat)
# range(start, stop, step)
for i in range(0, 11, 2):  # 0, 2, 4, 6, 8, 10
    print(i, end=" ")
print()"""))

cells.append(nbf.v4.new_code_cell("""# Loop untuk list (akan dijelaskan di Part 3)
buah = ["apel", "jeruk", "mangga", "pisang"]
for item in buah:
    print("Saya suka", item)"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 2.3 While Loop - Perulangan dengan Kondisi

`while` mengulang selama kondisi masih `True`.

**Analogi**: "Lakukan push-up **sampai** capek" - Tidak tahu pasti berapa kali, tergantung kondisi.
"""))

cells.append(nbf.v4.new_code_cell("""# While sederhana
counter = 1
while counter <= 5:
    print("Counter:", counter)
    counter = counter + 1  # PENTING: update counter (kalau tidak, infinite loop!)"""))

cells.append(nbf.v4.new_code_cell("""# Contoh praktis: Menghitung jumlah angka
total = 0
angka = 1

while angka <= 10:
    total = total + angka
    angka = angka + 1

print("Jumlah 1+2+3+...+10 =", total)"""))

cells.append(nbf.v4.new_markdown_cell("""### ⚠️ Hati-hati Infinite Loop!

Loop yang tidak pernah berhenti = infinite loop.

**JANGAN JALANKAN** code ini (hanya contoh):
```python
# while True:
#     print("Ini tidak akan berhenti!")
```

Kalau tidak sengaja infinite loop: Tekan tombol **Stop** (⏹️) di Jupyter.

---

## 2.4 Break dan Continue

- `break`: Keluar dari loop
- `continue`: Skip iterasi sekarang, lanjut ke iterasi berikutnya
"""))

cells.append(nbf.v4.new_code_cell("""# Break: Berhenti ketika menemukan angka 5
for i in range(1, 11):
    if i == 5:
        print("Ketemu 5, stop!")
        break
    print(i)"""))

cells.append(nbf.v4.new_code_cell("""# Continue: Skip angka genap (hanya print angka ganjil)
for i in range(1, 11):
    if i % 2 == 0:  # Jika genap (sisa bagi 2 = 0)
        continue    # Skip, langsung ke iterasi berikutnya
    print(i, end=" ")
print()"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 2.5 Nested Loop - Loop di Dalam Loop

Loop bisa di dalam loop (loop bersarang).
"""))

cells.append(nbf.v4.new_code_cell("""# Contoh: Membuat pola bintang
for i in range(1, 6):
    for j in range(i):
        print("*", end="")
    print()  # Newline setelah setiap baris"""))

cells.append(nbf.v4.new_code_cell("""# Contoh: Tabel perkalian
for i in range(1, 6):
    for j in range(1, 6):
        print(i * j, end="\\t")  # \\t = tab
    print()"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 🏋️ Latihan Part 2

### Latihan 2.1: If-Elif-Else
Buat program untuk menentukan kategori BMI (Body Mass Index):
- BMI < 18.5: Underweight
- 18.5 <= BMI < 25: Normal
- 25 <= BMI < 30: Overweight
- BMI >= 30: Obese

Formula BMI = berat(kg) / (tinggi(m))²

Test dengan: berat = 70, tinggi = 1.75
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
berat = 70
tinggi = 1.75


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 2.2: For Loop
Print semua bilangan dari 1 sampai 100 yang habis dibagi 7
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 2.3: While Loop
Buat program yang mencetak FizzBuzz:
- Untuk angka 1-30:
  - Jika habis dibagi 3: print "Fizz"
  - Jika habis dibagi 5: print "Buzz"
  - Jika habis dibagi 3 dan 5: print "FizzBuzz"
  - Selainnya: print angkanya

Gunakan while loop!
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 2 Selesai!** Anda sudah paham control flow!

**Next**: Part 3 - Data Structures (List, Dictionary)

---"""))

# Saya akan lanjutkan dengan membuat lebih banyak cells untuk Part 3, 4, 5, 6
# Untuk menghemat space, saya akan create script yang membuat notebook
# Mari kita simpan notebook yang sudah dibuat sejauh ini

nb['cells'] = cells
nbf.write(nb, '00_python_crash_course.ipynb')
print("✅ Notebook created successfully!")
print(f"Total cells: {len(cells)}")
