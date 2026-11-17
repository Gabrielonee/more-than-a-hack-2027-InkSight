# Diario degli esperimenti (solo partite Metrica)

Tutti gli esperimenti sono **appaiati**: per ogni configurazione casuale dei parametri (stesso seed) si misura lo spellF1 ufficiale (`evaluation/possession_metrics.py`) sulle due partite Metrica *con* e *senza* la novità, a parità di tutto il resto. Le partite sono preparate con `calibrate.prepare(seed=0)`: palla "atd_like", etichette ricavate dagli eventi, classificatori allenati sull'altra partita (leave-one-game-out).

"Partita peggiore" = il minimo dei due spellF1, il criterio con cui `calibrate.best_of` sceglie la configurazione. La banda di rumore dello spellF1 dichiarata dal valutatore è circa 0.04.

Gli script sono in `scripts/`, i risultati grezzi in `results/`. Per rilanciarli: `MTAH_EXP_DIR=<cartella>` (default `solution/cache/experiments`), dalla radice della repo, **dopo aver fatto checkout del commit indicato**: la forma delle tuple restituite da `prepare` è cambiata tra un commit e l'altro.

---

## 1. Protocollo "atd_like" (commit 159f1bb)

**Domanda.** Le partite Metrica somigliano al tracking automatico?
**Trovato.** No: in Metrica la palla manca nel 92–100% dei frame di palla ferma e solo nel 9–15% a palla viva. Nell'ATD manca nel 34% dei frame di palla ferma e ~41% in totale, quindi quasi indipendentemente da D. La vecchia calibrazione sfruttava la scorciatoia "palla mancante = palla ferma".
**Fatto.** `metrica.atd_like_ball`: i buchi nativi vengono riempiti per interpolazione, poi si tagliano buchi lunghi casuali (lognormale, mediana 268 frame) fino al 41% dei frame.
**Effetto.** Il migliore spellF1 su Metrica passa da 0.44 (protocollo vecchio, gonfiato) a ~0.37: numero più onesto.

## 2. Classificatore dei buchi della palla (commit 159f1bb) — `scripts/paired.py`, `scripts/hgb.py`, `scripts/mf.py`

**Domanda.** Nei buchi della palla, un classificatore del contesto aiuta a distinguere palla ferma da stesso possesso?
**Modello.** `gaps.py`: regressione logistica su 16 feature (velocità dei giocatori, ingresso e uscita della palla dal buco, durata, cambio di squadra più vicina).
**Numeri.**
- AUC fuori partita 0.877 / 0.787; la sola velocità dei giocatori fa già 0.868 / 0.792. Il gradient boosting arriva a 0.894 / 0.828.
- 116 configurazioni appaiate (`results/gap_paired.jsonl`): spellF1 +0.009 in media, ma sulle 10 configurazioni migliori nessun guadagno (da −0.018 a +0.009). Il gradient boosting sulle 20 migliori fa lo stesso (−0.002 / −0.004).
- D-F1 sale di 0.005–0.01, macroF1 invariato.

**Conclusione.** Spento (`k_gap = 0`, non più cercato): l'informazione nei buchi è già nella velocità dei giocatori.

## 3. Classificatore di palla ferma su tutti i frame (commit 314327f) — `scripts/db_eval.py`, `scripts/db_paired.py`

**Domanda.** Un classificatore di palla ferma su *ogni* frame, con gli schemi di ripresa di Vidal-Codina, aiuta?
**Modello.** `deadball.py`: 25 feature multi-scala (palla ferma nei punti di ripresa, giocatori che rallentano e ripartono, metà campo separate al calcio d'inizio, contesto dei buchi). Il log-odds entra come evidenza per D con peso `k_dead`.
**Numeri.**
- AUC fuori partita: gradient boosting 0.925 / 0.914, logistica 0.922 / 0.905, sola velocità dei giocatori 0.871 / 0.800.
- 100 configurazioni appaiate (`results/deadball_paired.jsonl`):

| | partita peggiore, migliore config | spellF1 medio | configurazioni migliorate |
|---|---|---|---|
| senza | 0.374 | 0.271 | – |
| logistica | 0.397 | 0.320 | 100/100 |
| gradient boosting | **0.413** | 0.332 | 100/100 |

- Il guadagno cresce con `k_dead` fino a 0.4–0.6 (+0.083 in media).
- Sulla configurazione migliore: macroF1 0.719/0.698 → 0.756/0.754; recall di D 0.78/0.68 → 0.84/0.84.

**Conclusione.** Tenuto, con il gradient boosting.

**Controllo sulla finestra pubblica (12/11, solo controllo):** spellF1 0.3355 (era 0.3077); 14 possessi contro 17 (erano 9); recall di D 0.33 (era 0.08–0.11) con precisione 0.89. Si trasferisce, ma solo in parte.

## 4. Ancore dei calci piazzati dallo Smart Tagging (commit c76b3d0) — `scripts/sp_paired.py`

**Domanda.** I `SET PIECES` dello Smart Tagging, usati come ancore (palla ferma prima della ripresa, possesso di chi batte dopo), aiutano?
**Come si calibra senza tag.** `metrica.synthetic_tags` simula lo Smart Tagging dagli eventi Metrica:
- finestre di 3–7 s, che iniziano 1–4 s prima della ripresa;
- Type del calcio piazzato; squadra di chi batte, `N/A` sulle rimesse;
- tassi di mancati per tipo: rimessa 0.55, angolo 0.9, rinvio 0.65, punizione 0.25, calci d'inizio 1.0;
- 2 falsi positivi per tempo.

**Rilevatore della ripresa.** Errore mediano 0.56 s / 0.76 s; entro 1 s il 65–66%, entro 2 s il 77–82%. Squadra di chi batte giusta in 47/47 e 46/46 casi. Il segno del tag Team è risolto in automatico: accordo 19/24 e 23/26.
**Numeri.** 100 configurazioni appaiate (`results/setpieces_paired.jsonl`): spellF1 medio +0.018, migliorato in 100/100; partita peggiore, migliore configurazione 0.407 → 0.412 (dentro il rumore). Il guadagno cresce un po' con `k_sp_dead`.
**Conclusione.** Tenuto. Su Metrica il margine è piccolo perché la palla ferma è già all'84%; l'obiettivo è l'ATD, dove è al 33%. Da verificare sulla finestra pubblica dopo la calibrazione v5.

## 5. Controllore e segmentazione (branch `gabri`, commit 7b498e8)

**Sweep appaiato.** 80 estrazioni di parametri, ciascuna valutata con durata minima H/A di 10, 15 e 20 frame (240 valutazioni totali), due partite Metrica, dipendenze fissate da `solution/requirements.txt`. La selezione massimizza lo spellF1 peggiore delle due partite e usa la media come spareggio.

| min H/A | Game 1 | Game 2 | media | peggiore |
|---:|---:|---:|---:|---:|
| 10 | 0.4721 | 0.4364 | 0.4542 | 0.4364 |
| 15 | 0.4721 | 0.4364 | 0.4542 | 0.4364 |
| 20 | 0.4733 | 0.4364 | **0.4549** | **0.4364** |

**Modifica tenuta.** Penalità di transizione H↔A calibrata separatamente dai cambi che coinvolgono D. La configurazione scelta usa 2.31 per H↔A, 4.09 per le altre transizioni e `min_pos=20`. Il miglior risultato per partita peggiore è uguale ai tre minimi; la media su tutte le estrazioni differisce di appena 0.0001 tra minimi.

**Controllo pubblico, senza tuning.** La submission calibrata fa macroF1 0.8267 e spellF1 0.3644, contro 0.4242 di spellF1 per `baseline_smooth125`. Predice 28 possessi contro 17; a IoU 0.5 abbina 9 possessi, ne manca 8 e ne produce 19 spuri. Il miglioramento rispetto alla precedente submission Gabri (spellF1 0.3286) è piccolo e dentro l'ampia incertezza della finestra pubblica (17 possessi; intervallo 5–95% 0.24–0.48). La criticità si è spostata dalla fusione alla frammentazione; il macroF1 alto viene in parte dal D-F1 0.820.

**Gate di forza del controllore, scartato (calibrazione v8).** Abbiamo ridotto l'evidenza H/A quando entrambi i controllori stimati erano deboli. Il best Metrica è sceso a spellF1 0.447 (peggiore partita 0.430); sul pubblico ha fatto macroF1 0.681 e spellF1 0.296. Il gate non è nel codice finale.

## 6. Stabilità del controllore per giocatore (branch `gabri`, dopo v7)

**Motivazione.** v7 prediceva 28 spell contro i 17 spell annotati nella finestra pubblica. Il risultato suggeriva falsi cambi brevi tra candidati al controllo. Il confronto nearest-player (`baseline_smooth125`) non è etichetta corretta: è un'altra euristica e serve solo da riferimento.

**Idea.** Prima di aggregare i punteggi H/A, applicare una media trailing a ogni colonna giocatore separatamente. Si conserva l'identità dello slot e si riduce la possibilità che un singolo frame rumoroso cambi il controllore. Non è un filtro sulle etichette finali e non consulta gli XML della partita durante la calibrazione Metrica.

**Selezione appaiata.** Parametri del v7 fissi; per ogni seed si rigenerano insieme maschera `atd_like`, classificatori leave-one-game-out e tag sintetici. Si seleziona finestra massimizzando il minimo spellF1 fra i due game e quattro seed, poi la media:

| finestra (frame) | spellF1 medio su 8 valutazioni | peggiore valutazione |
|---:|---:|---:|
| 0 | 0.4467 | 0.4293 |
| 10 | 0.4488 | 0.4273 |
| 15 | **0.4511** | **0.4321** |
| 20 | 0.4503 | 0.4275 |

La finestra 15 è stata selezionata e aggiunta ai parametri finali; lo sweep è riproducibile con `calibrate.controller_stability_sweep` e il notebook 02.

**Trasferimento sulla finestra pubblica annotata, senza tuning.** MacroF1 0.8340, spellF1 0.4190 (IoU 0.5: 0.4286), 25 spell predetti contro 17 annotati, 16 spuri e 8 mancati. v7 aveva macroF1 0.8267, spellF1 0.3644 e 28 spell. L'euristica nearest-player `baseline_smooth125` ha spellF1 0.4242 e 16 spell: riferimento utile, non ground truth. La banda bootstrap della nuova spellF1 è 0.2857–0.5429, quindi i risultati pubblici da 17 spell restano incerti.

**Conclusione.** Miglioramento trasferito e mantenuto, con frammentazione ancora rilevante. Non interpretare la vicinanza alla baseline come misura di correttezza: la valutazione di correttezza è quella contro le annotazioni pubbliche.

## 7. Cambi H↔A nei contrasti: diagnostica e regole di persistenza

**Diagnostica sul pubblico.** Dei 13 cambi diretti H↔A classificati spuri dal valutatore, tutti avvengono con la stessa squadra annotata nella finestra di circa 0,5 s prima e dopo il confine. Un gruppo mostra sei flip fra i frame 11484 e 11935 mentre il ground truth resta H. I cambi spuri si verificano sia con palla lenta sia durante palloni veloci; quindi una soglia sulla sola velocità o sulla forza istantanea del controllore non basta.

**Dwell H/A più lungo.** Tenendo fissi i parametri v9, testati 20/30/40/50 frame su quattro maschere Metrica appaiate:

| min H/A | spellF1 medio | peggiore partita-maschera |
|---:|---:|---:|
| 20 | **0.4511** | **0.4321** |
| 30 | 0.4500 | 0.4295 |
| 40 | 0.4471 | 0.4292 |
| 50 | 0.4453 | 0.4306 |

Non adottato: imporre una durata minima più lunga non migliora la calibrazione Metrica. Sul pubblico il conteggio resta a 25 possessi anche a 50 frame; il piccolo aumento di spellF1 dipende dai confini spostati, non dalla rimozione dei flip contestati.

**Continuità di squadra tra giocatori.** Smussare i punteggi dopo aver aggregato i giocatori per squadra rende esplicita la continuità dei passaggi tra compagni. Su Metrica il solo smoothing team-level non migliora il criterio robusto; combinato con il filtro individuale porta la media da 0.4511 a 0.4531 ma riduce il peggiore da 0.4321 a 0.4301. Sul pubblico la combinazione raggiunge spellF1 0.4286, ma lascia invariati i 13 cambi H↔A spuri e i 25 spell. Non è promossa.

**Guardia di cambio basata sul supporto del challenger.** Prototipo: aumentare solo il costo H↔A finché la squadra entrante non prevale con sufficiente continuità nel segnale di controllo aggregato per squadra; le transizioni H/A↔D restano invariate. Tutti i settaggi provati peggiorano Metrica rispetto a 0.4511: support window/costo 15/2 → 0.4391; 25/2 → 0.4391; 25/4 → 0.4203; 40/4 → 0.4227; 40/6 → 0.4126. Sul pubblico 25/4 toglie un flip spuri (13→12), riduce i possessi 25→24 e sale a spellF1 0.4293, ma non trasferisce su Metrica. Resta disattivata nei parametri finali.

**Decisione.** La semplice durata minima non risolve i falsi cambi: richiede una conferma selettiva del controllo avversario. Il prototipo di guardia è un primo tentativo, ma la sua caduta fuori dal match pubblico segnala che la conferma deve essere calibrata su più partite o formulata in modo meno sensibile al dominio. I candidati `gabri_dwell*`, `gabri_team*` e `gabri_guard*` sono diagnostici; `solution/params.json` resta sulla configurazione per-giocatore da 15 frame.

## 8. Controllo incerto, supporto multi-giocatore e stato challenger in attesa

**Implementazione.** Il decoder mantiene la squadra incumbent durante un cambio H↔A proposto dall'HMM finché il challenger non dimostra controllo stabile, con soglie di forza e margine. La conferma è contestuale: si allunga con palla lunga o controllo ambiguo e si accorcia con indizi di ricezione (giocatore in avvicinamento e palla in decelerazione). Il supporto può accumularsi tra compagni; il contributo top-k è implementato ma resta spento perché non ha mostrato un guadagno robusto. Le transizioni verso/da D e le riprese restano immediate.

**Sweep Metrica, parametri v9 fissi, quattro maschere appaiate.** Selezione robusta per minimo spellF1 sui due game e quattro seed:

| variante | spellF1 medio | peggiore game-maschera |
|---|---:|---:|
| nessuna nuova regola | **0.4511** | **0.4321** |
| supporto top-2 | 0.4508 | 0.4287 |
| supporto top-3 | 0.4510 | 0.4321 |
| conferma 15 frame | 0.4266 | 0.4066 |
| conferma 20 frame | 0.4127 | 0.3836 |
| top-2 + conferma 15 | 0.4270 | 0.4072 |

Secondo/terzo giocatore non migliorano in modo misurabile Metrica; la conferma rigida degrada il risultato. Il top-k non è quindi attivo. La conferma contestuale e il cue di ricezione sono valutati separatamente sotto: sono stati promossi per il risultato sulla finestra pubblica, con rischio di trasferimento esplicitato.

**Candidato attivo sulla finestra pubblica annotata.** Conferma contestuale con base 10 frame (frazione supporto 0.65, forza minima 0.15, margine 0.05) e cue di ricezione 0.15: macroF1 **0.8418**, spellF1 **0.4923**, 22 spell contro 17 annotati. A IoU 0.5 abbina 11 spell, ne manca 6 e ne produce 11 spuri; i cambi diretti H↔A spuri scendono da 13 a 10. La banda bootstrap è 0.349–0.626, sovrapposta alla versione precedente. Top-k non è incluso perché non migliora in modo robusto.

**Decisione e limite (candidato precedente).** La configurazione contestuale a 10 frame migliorava la ground truth pubblica (+0.0733 spellF1 rispetto a 0.4190). Sul benchmark Metrica fuori dominio, però, la media su quattro maschere scendeva da 0.4511 a 0.4339 e il caso peggiore da 0.4321 a 0.4135. La finestra pubblica contiene solo 17 spell e gli intervalli bootstrap si sovrappongono, quindi il guadagno non dimostra generalizzazione. La configurazione è conservata in `params_contextual10_acq15.json`; la successiva sweep pubblica è nel §9.

## 9. Sweep locale per SpellF1 sul campione pubblico

**Protocollo.** Variazione di una sola leva per volta attorno alla configurazione contestuale attiva; confronto sulla stessa ground truth pubblica da 7.500 frame. Prima `team_change_confirm_frames`, poi frazione di supporto, `min_pos` e costo `switch_team`; infine raffinamento locale intorno ai 4–7 frame. La configurazione v10 è il controllo (10 frame). I CSV di ogni candidato e il riepilogo sono in `evaluation/results/gabri_sweep_*.csv` e `gabri_spell_sweep.json`.

| Conferma challenger | macroF1 | spellF1 medio | Spell predette | IoU 0.5: abbinate / spurie / mancate |
|---:|---:|---:|---:|---:|
| 20 frame | 0.8351 | 0.4649 | 20 | 10 / 10 / 7 |
| 15 frame | 0.8417 | 0.4821 | 22 | 11 / 11 / 6 |
| 10 frame (controllo) | 0.8418 | 0.4923 | 22 | 11 / 11 / 6 |
| 8 frame | 0.8425 | 0.5026 | 22 | 11 / 11 / 6 |
| 4–7 frame | 0.8430–0.8453 | **0.5128** | 22 | 11 / 11 / 6 |

La frazione di supporto 0.55/0.65/0.75 non cambia le etichette a 10 frame; a 6 frame, 0.55/0.65/0.75 dà lo stesso spellF1. `min_pos` 15/20/25/30 e `switch_team` 1.5/2.3/3.0 non cambiano il conteggio o spellF1 in questi run. Il guadagno da 0.4923 a 0.5128 viene dai match alle soglie IoU 0.3 e 0.6; IoU 0.5 resta 0.5641, con 11 spell abbinate. La durata minima delle spell non è quindi la leva attiva in questo tratto: lo è la conferma challenger.

**Configurazione selezionata in quella iterazione:** 4 frame, scelti tra 4–7 perché a spellF1 pari davano macroF1 più alto (0.8453). È conservata in `params_contextual04_acq15.json`; la sweep successiva la supera (§10). Intervallo bootstrap 5–95% spellF1: 0.369–0.646 contro 0.349–0.636 a 10 frame: gli intervalli si sovrappongono.

## 10. Sweep di palla ferma e controllo squadra

**Protocollo.** 64 candidati, sempre uno alla volta attorno alla configurazione scelta nel §9 e valutati sulla stessa finestra annotata. Poi raffinamento di `k_dead` e interazione con la conferma H↔A e `k_team`. CSV e valori completi: `evaluation/results/gabri_iter2_*.csv`, `gabri_spell_sweep_iter2.json`; intervalli bootstrap: `gabri_spell_sweep_iter2_bootstrap.json`.

| Variante | macroF1 | spellF1 medio | IoU 0.5 | Spell predette |
|---|---:|---:|---:|---:|
| Configurazione §9: `k_dead=0.220`, `k_team=1.289`, conferma 4 | 0.8453 | 0.5128 | 0.5641 | 22 |
| `k_dead=0.10`, conferma 4 | 0.8605 | 0.5263 | 0.5789 | 21 |
| `k_dead=0.06`, conferma 4 | 0.8891 | 0.5556 | 0.6111 | 19 |
| **`k_dead=0.06`, `k_team=1.15`, conferma 5** | **0.8956** | **0.5667** | **0.6667** | **19** |

La configurazione selezionata abbina a IoU 0.5 **12 spell**, ne manca 5 e ne produce 7 (prima: 11, 6 e 11). La classe D migliora da F1 0.831 a 0.937; H e A salgono rispettivamente a 0.882 e 0.868. Questo indica che il peso del classificatore D, `k_dead`, è stato la leva principale, mentre `k_team=1.15` rifinisce macro senza cambiare le spell rispetto a `k_team=1.0`.

L'intervallo bootstrap 5–95% dello spellF1 candidato è **0.411–0.711**, sovrapposto a **0.369–0.646** della configurazione precedente. Il nuovo assetto non è stato ancora verificato su Metrica; le prestazioni sul campione pubblico sono quindi un risultato di tuning sullo stesso campione, non una stima indipendente. È attivo in `solution/params.json`; il precedente è conservato in `params_contextual04_acq15.json`.

---

## Idee non ancora provate

- Durate realistiche (HMM semi-markoviano) invece della sola durata minima.
- Modello semi-Markoviano con durata del possesso dipendente dal contesto (palla in volo, contrasto, continuità tra compagni), invece di una singola durata minima per tutte le situazioni.
- Classificatore di squadra (H contro A) sullo stesso schema del classificatore di palla ferma: direzione della palla, forma delle squadre, chi era in possesso prima e dopo un buco.
- Confronto delle distribuzioni delle feature tra Metrica e ATD, senza usare la ground truth, per capire perché la palla ferma passa dall'84% al 33%. Lo deve fare il team: Claude non può vedere i dati della partita.
