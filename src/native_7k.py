import math

def calculate_native_7k_sr(hitobjects, column_width=512/7):
    """
    Menghitung Star Rating dan mengklasifikasikan pola dominan (Skillsets)
    termasuk pendeteksi pola khusus 7K, Top 15% Average, dan Pattern Synthesis.
    """
    if not hitobjects:
        return {"overall": 0.0}

    hitobjects.sort(key=lambda obj: obj['time'])

    # --- Hitung ln_ratio global di awal ---
    total_objects = len(hitobjects)
    ln_objects = sum(1 for obj in hitobjects if obj.get('type', 1) & 128)
    ln_ratio = ln_objects / total_objects if total_objects > 0 else 0.0

    # Strain trackers
    current_strain = 0.0
    all_overall_strains = []  # List untuk menyimpan semua nilai strain note
    
    # Kategori Skillset Mentah
    # Strain trackers
    current_strain = 0.0
    all_overall_strains = []
    
    # GANTI INI: Kita bikin list kosong untuk nyimpen SEMUA nilai dari setiap note
    skill_strains = {
        "stamina": [],
        "jack": [],
        "chord": [],
        "ln": [],
        "stairs": [],
        "bracket": []
    }

    last_time = hitobjects[0]['time']
    last_columns = set()
    decay_base = 0.3

    # Kelompokkan objek (Chord/Notes dalam satu waktu)
    time_groups = {}
    ln_count_per_time = {}
    
    for obj in hitobjects:
        t = obj['time']
        col = int(obj['x'] // column_width)
        
        if t not in time_groups:
            time_groups[t] = []
            ln_count_per_time[t] = 0
            
        time_groups[t].append(col)
        
        # Cek apakah ini LN
        if obj.get('type', 1) & 128:
            ln_count_per_time[t] += 1

    sorted_times = sorted(time_groups.keys())

    # --- LOOP KALKULASI UTAMA ---
    for i in range(1, len(sorted_times)):
        t = sorted_times[i]
        cols = set(time_groups[t])
        is_ln = ln_count_per_time[t] > 0

        delta_t = (t - last_time) / 1000.0 
        if delta_t <= 0:
            continue

        
        speed_strain = 1.0 / max(delta_t, 0.05) 
        
        
        chord_size = len(cols)
        chord_bonus = (chord_size - 1) * 0.4
        
        
        jacks = len(cols.intersection(last_columns))
        jack_bonus = 0.0
        if jacks > 0:
            jack_bonus = (jacks ** 1.5) * 0.3

        # 4. Deteksi Bracket (Mengisi kekosongan pattern sebelumnya)
        bracket_bonus = 0.0
        if chord_size >= 2 and len(last_columns) >= 2 and jacks == 0:
            bracket_bonus = 1.2

        # 5. Deteksi Stairs (Pergerakan Tangga Berurutan)
        stairs_bonus = 0.0
        if chord_size == 1 and len(last_columns) == 1:
            col_now = list(cols)[0]
            col_prev = list(last_columns)[0]
            if abs(col_now - col_prev) == 1:
                stairs_bonus = 0.8

        # --- Kalkulasi Beban Total Note Ini ---
        note_strain = speed_strain * (1.0 + chord_bonus + jack_bonus + bracket_bonus + stairs_bonus)
        current_strain = current_strain * math.exp(-decay_base * delta_t) + note_strain

        # Simpan nilai strain ke dalam list untuk di-rata-rata nanti
        all_overall_strains.append(current_strain)

       # ... (kalkulasi note_strain dan current_strain tetap sama) ...
        all_overall_strains.append(current_strain)

        # GANTI INI: Masukkan nilai ke dalam list masing-masing
        skill_strains["stamina"].append(speed_strain)
        skill_strains["jack"].append(jack_bonus * speed_strain)
        skill_strains["chord"].append(chord_bonus * speed_strain)
        skill_strains["bracket"].append(bracket_bonus * speed_strain)
        skill_strains["stairs"].append(stairs_bonus * speed_strain)

        # Khusus LN, kalau bukan note LN kita kasih 0.0 biar rata-ratanya akurat
        if is_ln:
            skill_strains["ln"].append(current_strain)
        else:
            skill_strains["ln"].append(0.0)


        last_time = t
        last_columns = cols

    # --- JALAN TENGAH: TOP 15% AVERAGE ---
    if not all_overall_strains:
        return {"overall": 0.0}
        
    all_overall_strains.sort(reverse=True)
    top_15_count = max(1, int(len(all_overall_strains) * 0.15))
    average_peak = sum(all_overall_strains[:top_15_count]) / top_15_count

    # --- FUNGSI HELPER TOP 15% UNTUK SKILLSET ---
    def get_top_15_avg(strain_list):
        if not strain_list: return 0.0
        strain_list.sort(reverse=True)
        count = max(1, int(len(strain_list) * 0.15))
        return sum(strain_list[:count]) / count

    # --- SINTESIS POLA (TIERED SKILLSETS) ---
    p_stam = get_top_15_avg(skill_strains["stamina"])
    p_chord = get_top_15_avg(skill_strains["chord"])
    p_jack = get_top_15_avg(skill_strains["jack"])
    p_stairs = get_top_15_avg(skill_strains["stairs"])
    p_bracket = get_top_15_avg(skill_strains["bracket"])
    p_ln = get_top_15_avg(skill_strains["ln"])

    synthesized = {}

    # 1. RICE MAP (0% - 25% LN)
    if ln_ratio <= 0.25:
        synthesized["Chordstream"] = (p_stam * 0.6) + (p_chord * 0.4)
        synthesized["Chordjack"] = (p_jack * 0.6) + (p_chord * 0.4)
        synthesized["Stream"] = max(0.0, p_stam - (p_chord * 0.5))
        synthesized["Jack"] = max(0.0, p_jack - (p_chord * 0.5))
        synthesized["Stairs"] = p_stairs
        synthesized["Bracket"] = p_bracket

    # 2. HYBRID MAP (25% - 55% LN)
    elif 0.25 < ln_ratio <= 0.55:
        # Buka gerbang dua dunia: Rice dan LN bertarung secara adil di sini!
        # Patern Rice
        synthesized["Chordstream"] = (p_stam * 0.6) + (p_chord * 0.4)
        synthesized["Chordjack"] = (p_jack * 0.6) + (p_chord * 0.4)
        synthesized["Stream"] = max(0.0, p_stam - (p_chord * 0.5))
        synthesized["Jack"] = max(0.0, p_jack - (p_chord * 0.5))
        
        # Patern LN
        synthesized["LNstream"] = (p_ln * 0.5) + (p_stam * 0.5)
        synthesized["Inverse"] = (p_ln * 0.4) + (p_stam * 0.228) + (p_chord * 0.2) + (p_bracket * 0.1)
        
        # Pattern Netral
        synthesized["Stairs"] = p_stairs
        synthesized["Bracket"] = p_bracket

    # 3. FULL LN MAP (> 55% LN)
    else:
        # Rice dimatikan, murni surga buat pemain noodle/LN.
        synthesized["LNstream"] = (p_ln * 0.5) + (p_stam * 0.5)
        synthesized["Inverse"] = (p_ln * 0.4) + (p_stam * 0.228) + (p_chord * 0.2) + (p_bracket * 0.1)
        synthesized["Stairs"] = p_stairs
        synthesized["Bracket"] = p_bracket

    multiplier = 0.59 
    final_sr = (average_peak ** 0.5) * multiplier
    
    
    ui_floor = 0.825
    
    max_raw_skill = max(synthesized.values()) if max(synthesized.values()) > 0 else 1.0
    
    result = {"overall": final_sr}
    
    for pattern_name, raw_val in synthesized.items():
        if raw_val <= 0:
            result[pattern_name] = 0.0
        else:
            # Rumus peregangan skala (Stretching)
            ratio = raw_val / max_raw_skill
            ui_friendly_ratio = ui_floor + (ratio * (1.0 - ui_floor))
            
            result[pattern_name] = final_sr * ui_friendly_ratio
            
    return result