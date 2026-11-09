"""Parlama (specular highlight) tespiti ve maske olusturma yardimcilari."""
import cv2, numpy as np

def glare_mask(bgr, v_thresh=220, s_thresh=60):
    """Parlak + dusuk-doygunlukta pikselleri parlama olarak isaretler.
    Doner: 0/255 ikili maske."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((v >= v_thresh) & (s <= s_thresh)).astype(np.uint8) * 255
    # kucuk gurultuyu temizle, parlamayi biraz genislet
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask

def glare_ratio(bgr, **kw):
    """Goruntudeki parlama piksel oranini [0,1] doner."""
    m = glare_mask(bgr, **kw)
    return float((m > 0).mean())

def telea_inpaint(bgr, mask):
    """Klasik (GAN olmayan) referans parlama giderme: Telea inpainting."""
    return cv2.inpaint(bgr, (mask > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

def feather_mask(mask, blur=9, grow=2):
    """Ikili maskeyi biraz genislet + bulanik kenar -> yumusak [0,1] harman maskesi."""
    m = cv2.dilate((mask > 0).astype(np.uint8) * 255,
                   np.ones((3, 3), np.uint8), iterations=grow)
    m = cv2.GaussianBlur(m, (blur, blur), 0).astype(np.float32) / 255.0
    return m  # 0..1

def blend_deglare(original_bgr, gan_bgr):
    """Maskeli/hedefli parlama giderme:
    GAN ciktisini SADECE parlama bolgesinde uygula, kalan dokuyu orijinal birak.
    Boylece tanisal doku birebir korunur.
    original_bgr, gan_bgr: ayni boyutta BGR uint8.
    """
    m = feather_mask(glare_mask(original_bgr))            # H x W, 0..1
    m3 = m[:, :, None]
    out = original_bgr.astype(np.float32) * (1 - m3) + gan_bgr.astype(np.float32) * m3
    return out.clip(0, 255).astype("uint8")
