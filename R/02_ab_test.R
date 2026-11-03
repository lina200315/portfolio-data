# Projet : Faut-il lancer la nouvelle page de vente ? Analyse d'un test A/B
# Données : 294 478 visites d'un site e-commerce (janvier 2017), groupe contrôle vs traitement
# Méthode : contrôle qualité (doublons, incohérences) -> test de proportions -> intervalle de confiance
#           -> puissance statistique -> approche bayésienne -> suivi cumulé jour par jour

suppressMessages(library(jsonlite))
root <- if (dir.exists("data")) "." else ".."
df <- read.csv(file.path(root, "data/ab_test.csv"), stringsAsFactors = FALSE)

# ---------- 1. Qualité des données ----------
n_brut <- nrow(df)
incoh <- with(df, (group == "control" & landing_page == "new_page") |
                  (group == "treatment" & landing_page == "old_page"))
n_incoh <- sum(incoh)
df <- df[!incoh, ]
n_dup <- sum(duplicated(df$user_id))
df <- df[!duplicated(df$user_id), ]
df$jour <- as.Date(substr(df$timestamp, 1, 10))

# Test de ratio d'échantillon (SRM) : la répartition 50/50 est-elle respectée ?
tab <- table(df$group)
srm_p <- chisq.test(tab, p = c(.5, .5))$p.value

# ---------- 2. Test fréquentiste ----------
conv <- tapply(df$converted, df$group, sum)
n <- tapply(df$converted, df$group, length)
rate <- conv / n
pt <- prop.test(conv[c("treatment", "control")], n[c("treatment", "control")], correct = FALSE)
diff <- rate["treatment"] - rate["control"]
uplift <- diff / rate["control"] * 100

# ---------- 3. Puissance : quel effet pouvait-on détecter ? ----------
mde <- power.prop.test(n = min(n), p1 = rate["control"], power = 0.8, sig.level = 0.05)$p2 - rate["control"]
# Taille nécessaire pour détecter +2 % relatif
n_needed <- ceiling(power.prop.test(p1 = rate["control"], p2 = rate["control"] * 1.02,
                                    power = 0.8, sig.level = 0.05)$n)

# ---------- 4. Approche bayésienne (a priori Beta(1,1)) ----------
set.seed(2017)
S <- 200000
pa <- rbeta(S, 1 + conv["control"], 1 + n["control"] - conv["control"])
pb <- rbeta(S, 1 + conv["treatment"], 1 + n["treatment"] - conv["treatment"])
p_b_better <- mean(pb > pa)
lift_post <- (pb - pa) / pa * 100
h <- hist(lift_post, breaks = seq(floor(min(lift_post)), ceiling(max(lift_post)), by = 0.25), plot = FALSE)

# ---------- 5. Suivi cumulé : le danger de regarder trop tôt ----------
days <- sort(unique(df$jour))
cum <- do.call(rbind, lapply(days, function(d) {
  s <- df[df$jour <= d, ]
  ca <- sum(s$converted[s$group == "control"]); na <- sum(s$group == "control")
  cb <- sum(s$converted[s$group == "treatment"]); nb <- sum(s$group == "treatment")
  pv <- prop.test(c(cb, ca), c(nb, na), correct = FALSE)$p.value
  data.frame(jour = format(d, "%d/%m"), a = ca / na * 100, b = cb / nb * 100, p = pv)
}))
daily <- aggregate(converted ~ jour + group, df, mean)

# Segment : effet par jour de semaine (vérifie l'absence d'effet caché)
df$wd <- factor(weekdays(df$jour), levels = c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))
wd <- aggregate(converted ~ wd + group, df, mean)
wd_fr <- c("Lun","Mar","Mer","Jeu","Ven","Sam","Dim")

out <- list(
  qualite = list(brut = n_brut, incoherents = n_incoh, doublons = n_dup, final = nrow(df), srm_p = round(srm_p, 3)),
  groupes = list(n_a = unname(n["control"]), n_b = unname(n["treatment"]),
                 conv_a = round(unname(rate["control"]) * 100, 3), conv_b = round(unname(rate["treatment"]) * 100, 3)),
  test = list(diff_pts = round(unname(diff) * 100, 3), uplift = round(unname(uplift), 2), p = round(pt$p.value, 3),
              ic_bas = round(pt$conf.int[1] * 100, 3), ic_haut = round(pt$conf.int[2] * 100, 3)),
  puissance = list(mde_pts = round(unname(mde) * 100, 3), mde_rel = round(unname(mde / rate["control"]) * 100, 2),
                   n_pour_2pct = n_needed),
  bayes = list(p_b_meilleur = round(p_b_better * 100, 1), lift_q05 = round(quantile(lift_post, .05), 2),
               lift_q95 = round(quantile(lift_post, .95), 2),
               hist = list(x = head(h$breaks, -1) + 0.125, y = h$counts)),
  cumul = cum,
  semaine = list(jours = wd_fr, a = round(wd$converted[wd$group == "control"] * 100, 2),
                 b = round(wd$converted[wd$group == "treatment"] * 100, 2))
)
write_json(out, file.path(root, "results/r_abtest.json"), auto_unbox = TRUE, digits = 6, pretty = TRUE)
str(out[1:5])
print(cum[, c("jour", "p")])
