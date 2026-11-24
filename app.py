
import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageOps
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi OCR AI",
    page_icon="📄",
    layout="centered"
)

# Judul aplikasi
st.title("📄 Aplikasi OCR Berbasis AI")
st.write(
    "Aplikasi untuk mengenali dan mengekstrak "
    "teks dari gambar secara otomatis menggunakan AI."
)

st.info(
    "Unggah gambar yang berisi tulisan, "
    "kemudian tekan tombol Mulai OCR."
)

# Membuat model OCR hanya satu kali
@st.cache_resource
def load_ocr_model():
    reader = easyocr.Reader(
        ['id', 'en'],
        gpu=False
    )
    return reader

# Upload gambar
uploaded_file = st.file_uploader(
    "Pilih gambar:",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Membaca gambar
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Menampilkan gambar
    st.subheader("Gambar yang diunggah")
    st.image(
        image,
        caption="Gambar sumber",
        use_container_width=True
    )

    # Tombol OCR
    if st.button("🔍 Mulai OCR", type="primary"):

        try:
            with st.spinner(
                "AI sedang mengenali teks dalam gambar..."
            ):

                # Memuat model AI
                reader = load_ocr_model()

                # Mengubah gambar menjadi array
                image_array = np.array(image)

                # Proses pengenalan teks
                results = reader.readtext(
                    image_array,
                    detail=1,
                    paragraph=False
                )

            st.subheader("Hasil Pengenalan Teks")

            if results:

                extracted_text = []

                for index, result in enumerate(results, 1):

                    bounding_box, text, confidence = result

                    extracted_text.append(text)

                    st.write(
                        f"**{index}. {text}**"
                    )

                    st.caption(
                        f"Tingkat keyakinan AI: "
                        f"{confidence * 100:.2f}%"
                    )

                # Menggabungkan seluruh teks
                final_text = "\n".join(extracted_text)

                st.success(
                    f"Berhasil mengenali "
                    f"{len(extracted_text)} bagian teks."
                )

                # Menampilkan hasil dalam kotak teks
                st.text_area(
                    "Teks hasil ekstraksi:",
                    value=final_text,
                    height=200
                )

                # Tombol download
                st.download_button(
                    label="⬇️ Download Hasil TXT",
                    data=final_text,
                    file_name="hasil_ocr.txt",
                    mime="text/plain"
                )

            else:
                st.warning(
                    "Tidak ditemukan teks dalam gambar."
                )

        except Exception as error:
            st.error(
                f"Terjadi kesalahan: {error}"
            )

st.divider()

st.caption(
    "Proyek OCR AI | Python + Streamlit + EasyOCR"
)
