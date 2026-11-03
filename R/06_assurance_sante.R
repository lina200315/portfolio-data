# Projet : Tarification en assurance santé — qu'est-ce qui fait le coût d'un assuré ?
# Données : Medical Cost Personal Dataset (1 338 assurés américains : âge, sexe, IMC, enfants, tabac, région, frais médicaux annuels),
#           jeu pédagogique du livre « Machine Learning with R » (Brett Lantz)
# Méthode : distribution très asymétrique des coûts -> effet de chaque facteur
#           -> régression linéaire, puis avec interaction tabac × obésité -> validation croisée
#           -> grille tarifaire lisible (prime pure par profil)

suppressMessages(library(jsonlite))
root <- if (dir.exists("data")) "." else ".."
d <- read.csv(file.path(root, "data/insurance.csv"), stringsAsFactors = TRUE)
d$obese <- factor(ifelse(d$bmi >= 30, "Obèse (IMC ≥ 30)", "IMC < 30"))
d$age2 <- d$age^2

# ---------- 1. Exploration ----------
h <- hist(d$charges / 1000, breaks = seq(0, 66, 3), plot = FALSE)
grp <- function(v) { a <- aggregate(charges ~ g, data.frame(g = v, charges = d$charges), function(x) c(moy = mean(x), med = median(x), n = length(x)))
  data.frame(groupe = as.character(a$g), moyenne = round(a$charges[, "moy"]), mediane = round(a$charges[, "med"]), n = a$charges[, "n"]) }
d$tranche_age <- cut(d$age, c(17, 29, 39, 49, 59, 65), labels = c("18-29", "30-39", "40-49", "50-59", "60-64"))
facteurs <- list(Tabac = grp(d$smoker), `Tranche d'âge` = grp(d$tranche_age), Obésité = grp(d$obese), Sexe = grp(d$sex),
                 Région = grp(d$region), `Enfants à charge` = grp(pmin(d$children, 3)))
set.seed(4)
pts <- d[sample(nrow(d), 700), ]
scatter <- data.frame(age = pts$age, cout = round(pts$charges), tabac = as.character(pts$smoker), obese = as.character(pts$obese))

# ---------- 2. Modèles ----------
f1 <- charges ~ age + sex + bmi + children + smoker + region
f2 <- charges ~ age + age2 + sex + children + region + smoker * obese + bmi
m1 <- lm(f1, data = d); m2 <- lm(f2, data = d)
cv <- function(f, k = 10) { set.seed(1); fold <- sample(rep(1:k, length.out = nrow(d))); e <- numeric(nrow(d))
  for (i in 1:k) { m <- lm(f, data = d[fold != i, ]); e[fold == i] <- d$charges[fold == i] - predict(m, d[fold == i, ]) }
  c(rmse = sqrt(mean(e^2)), mae = mean(abs(e)), r2 = 1 - sum(e^2) / sum((d$charges - mean(d$charges))^2)) }
perf <- rbind(`Régression simple` = cv(f1), `Avec interaction tabac × obésité` = cv(f2))
co <- summary(m2)$coefficients
coefs <- data.frame(var = rownames(co), effet = round(co[, 1]), p = signif(co[, 4], 2))[-1, ]
rownames(coefs) <- NULL
# Résidus du modèle simple pour les fumeurs obèses : le modèle sans interaction les sous-estime
d$res1 <- resid(m1); d$res2 <- resid(m2)
bias <- aggregate(cbind(res1, res2) ~ smoker + obese, d, mean)
bias$res1 <- round(bias$res1); bias$res2 <- round(bias$res2)

# ---------- 3. Grille tarifaire (prime pure prédite, homme, 1 enfant, région nord-est) ----------
grid <- expand.grid(age = c(25, 35, 45, 55, 64), smoker = c("no", "yes"), obese = levels(d$obese))
grid$bmi <- ifelse(grid$obese == "IMC < 30", 25, 33); grid$age2 <- grid$age^2
grid$sex <- "male"; grid$children <- 1; grid$region <- "northeast"
grid$prime <- round(predict(m2, grid))

out <- list(
  kpi = list(assures = nrow(d), cout_moy = round(mean(d$charges)), cout_med = round(median(d$charges)),
             ratio_tabac = round(mean(d$charges[d$smoker == "yes"]) / mean(d$charges[d$smoker == "no"]), 1),
             part_fumeurs = round(mean(d$smoker == "yes") * 100, 1), r2 = round(perf[2, "r2"], 3), r2_simple = round(perf[1, "r2"], 3),
             part_top10 = round(sum(sort(d$charges, decreasing = TRUE)[1:round(nrow(d) * .1)]) / sum(d$charges) * 100, 1)),
  hist = list(x = head(h$breaks, -1), y = h$counts), facteurs = facteurs, scatter = scatter,
  perf = data.frame(modele = rownames(perf), round(perf, 3)), coefs = coefs, bias = bias, grille = grid[, c("age", "smoker", "obese", "prime")]
)
write_json(out, file.path(root, "results/r_assurance.json"), auto_unbox = TRUE, digits = 6, pretty = FALSE)
print(out$kpi); print(perf); print(coefs); print(bias); print(head(grid[grid$smoker == "yes", c("age", "obese", "prime")], 10))
