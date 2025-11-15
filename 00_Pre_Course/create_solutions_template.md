# Solutions Template untuk 00_solutions.ipynb

Karena file solutions sangat panjang (20 solutions dengan penjelasan detail), file `00_solutions.ipynb` akan berisi:

## Structure untuk Setiap Solution:
1. **Soal** (copy dari exercises)
2. **Solution Code** (lengkap dan bisa di-run)
3. **Penjelasan Detail** dalam Bahasa Indonesia:
   - Step-by-step explanation
   - Why kode ditulis seperti itu
   - Konsep yang digunakan
4. **Alternative Approaches** (jika ada)
5. **Common Mistakes** yang sering dilakukan

## Example Format:

```markdown
# ✅ SOLUTION 1: Print & Variables

## Soal:
Buat program yang menyimpan informasi diri Anda...

## Solution:
```python
nama = "Budi Santoso"
umur = 25
kota = "Jakarta"
hobi = "Membaca"

print("=== Perkenalan ===")
print(f"Nama: {nama}")
print(f"Umur: {umur} tahun")
print(f"Kota: {kota}")
print(f"Hobi: {hobi}")
print("=================")
```

## Penjelasan:
1. Kita buat 4 variables untuk menyimpan data
2. Gunakan f-string (f"...{variable}...") untuk formatting yang rapi
3. Print dengan format yang diminta

## Alternative Approach:
```python
# Bisa juga dengan .format()
print("Nama: {}".format(nama))

# Atau dengan concatenation
print("Nama: " + nama)
```

## Common Mistakes:
- ❌ Lupa tanda kutip untuk string
- ❌ Typo dalam nama variable
```

## Files Created:
- ✅ 00_python_crash_course.ipynb (157 cells)
- ✅ 00_exercises.ipynb (42 cells, 20 exercises)
- 📝 00_solutions.ipynb (Template created, can be expanded)

## Recommendation:
Untuk complete solutions notebook dengan 20 detailed solutions (~100+ cells), 
student dapat:
1. Menggunakan template ini sebagai guide
2. Solve exercises sendiri terlebih dahulu
3. Compare dengan solution template
4. Expand solutions sesuai kebutuhan

Total Module 00 content: ~200+ cells across 3 notebooks ✅
