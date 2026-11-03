# Projet : Évaluer l'impact d'un programme de formation sans essai randomisé — le problème de LaLonde (1986)
# Données : NSW (National Supported Work, États-Unis, années 1970) : 185 bénéficiaires de la formation,
#   - comparés à 429 non-bénéficiaires issus d'une enquête (PSID) -> données observationnelles (MatchIt::lalonde)
#   - et, pour vérifier, au groupe témoin tiré au sort de l'expérience originale (causaldata::nsw_mixtape)
# Question : de combien la formation a-t-elle augmenté le revenu de 1978 des bénéficiaires (effet moyen sur les traités, ATT) ?
# Méthode : comparaison naïve -> régression -> score de propension (glm logit) -> appariement 1:1 au plus proche voisin
#           avec remise et caliper -> pondération inverse (IPW) -> équilibre des covariables (SMD) -> bootstrap

suppressMessages(library(jsonlite))
root <- if (dir.exists("data")) "." else ".."
obs <- read.table(file.path(root, "data/lalonde_observationnel.tsv"), header = TRUE)
exp_ <- read.csv(file.path(root, "data/nsw_experimental.csv"))
obs$noir <- as.integer(obs$race == "black"); obs$hisp <- as.integer(obs$race == "hispan")
obs$u74 <- as.integer(obs$re74 == 0); obs$u75 <- as.integer(obs$re75 == 0)
X <- c("age", "educ", "noir", "hisp", "married", "nodegree", "re74", "re75", "u74", "u75")

# ---------- 0. La référence : l'expérience tirée au sort ----------
bench <- with(exp_, mean(re78[treat == 1]) - mean(re78[treat == 0]))
bench_se <- with(exp_, sqrt(var(re78[treat == 1]) / sum(treat == 1) + var(re78[treat == 0]) / sum(treat == 0)))

# ---------- 1. Naïf et régression ----------
naif <- with(obs, mean(re78[treat == 1]) - mean(re78[treat == 0]))
reg <- coef(lm(re78 ~ treat + age + I(age^2) + educ + noir + hisp + married + nodegree + re74 + re75, data = obs))["treat"]

# ---------- 2. Score de propension ----------
ps_f <- treat ~ age + I(age^2) + educ + I(educ^2) + noir + hisp + married + nodegree + re74 + re75 + u74 + u75
ps <- fitted(glm(ps_f, data = obs, family = binomial))
obs$ps <- ps; obs$lps <- qlogis(pmin(pmax(ps, 1e-6), 1 - 1e-6))

match_att <- function(d) {
  t <- which(d$treat == 1); c <- which(d$treat == 0)
  cal <- 0.2 * sd(d$lps)                                      # caliper usuel : 0,2 écart-type du logit
  m <- sapply(t, function(i) { dd <- abs(d$lps[c] - d$lps[i]); j <- which.min(dd); if (dd[j] <= cal) c[j] else NA })
  ok <- !is.na(m)
  list(att = mean(d$re78[t[ok]] - d$re78[m[ok]]), t = t[ok], c = m[ok], n_out = sum(!ok))
}
mt <- match_att(obs)

# ---------- 3. Pondération inverse (ATT : poids ps/(1-ps) pour les témoins) ----------
w <- ifelse(obs$treat == 1, 1, obs$ps / (1 - obs$ps))
ipw <- with(obs, sum(w * re78 * treat) / sum(w * treat) - sum(w * re78 * (1 - treat)) / sum(w * (1 - treat)))

# ---------- 4. Équilibre des covariables (différence moyenne standardisée) ----------
smd <- function(x, t, wt = rep(1, length(x))) {
  m1 <- weighted.mean(x[t == 1], wt[t == 1]); m0 <- weighted.mean(x[t == 0], wt[t == 0])
  (m1 - m0) / sqrt((var(x[t == 1]) + var(x[t == 0])) / 2) }
mt_idx <- c(mt$t, mt$c); mt_t <- c(rep(1, length(mt$t)), rep(0, length(mt$c)))
bal <- data.frame(var = X,
  avant = round(sapply(X, function(v) smd(obs[[v]], obs$treat)), 3),
  apres_appariement = round(sapply(X, function(v) smd(obs[[v]][mt_idx], mt_t)), 3),
  apres_ponderation = round(sapply(X, function(v) smd(obs[[v]], obs$treat, w)), 3),
  moy_traites = round(sapply(X, function(v) mean(obs[[v]][obs$treat == 1])), 2),
  moy_temoins = round(sapply(X, function(v) mean(obs[[v]][obs$treat == 0])), 2))

# ---------- 5. Incertitude : bootstrap de l'appariement ----------
set.seed(2024)
boots <- replicate(300, { b <- obs[sample(nrow(obs), replace = TRUE), ]
  b$ps <- fitted(suppressWarnings(glm(ps_f, data = b, family = binomial))); b$lps <- qlogis(pmin(pmax(b$ps, 1e-6), 1 - 1e-6))
  match_att(b)$att })
ci <- quantile(boots, c(.025, .975))

# Recouvrement des scores (support commun)
br <- seq(0, 1, by = 0.05)
h1 <- hist(obs$ps[obs$treat == 1], breaks = br, plot = FALSE)$counts
h0 <- hist(obs$ps[obs$treat == 0], breaks = br, plot = FALSE)$counts

out <- list(
  kpi = list(traites = sum(obs$treat), temoins = sum(obs$treat == 0), bench = round(bench), bench_se = round(bench_se),
             naif = round(naif), reg = round(unname(reg)), match = round(mt$att), ipw = round(ipw),
             ci_bas = round(ci[1]), ci_haut = round(ci[2]), hors_support = mt$n_out, temoins_utilises = length(unique(mt$c))),
  estimations = data.frame(methode = c("Expérience tirée au sort (référence)", "Comparaison naïve", "Régression linéaire", "Appariement sur le score", "Pondération inverse (IPW)"),
                           att = round(c(bench, naif, unname(reg), mt$att, ipw))),
  balance = bal, ps = list(x = head(br, -1) + 0.025, traites = h1, temoins = h0), boot = round(boots)
)
write_json(out, file.path(root, "results/r_impact.json"), auto_unbox = TRUE, digits = 6, pretty = FALSE)
print(out$kpi); print(out$estimations); print(bal)
