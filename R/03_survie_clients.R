# Projet : Combien de temps restent les clients ? Analyse de survie du churn télécom
# Données : IBM Telco Customer Churn (7 043 abonnés) — ancienneté en mois + résiliation oui/non
# Méthode : courbes de Kaplan-Meier par contrat et par service -> test du log-rank
#           -> modèle de Cox (hazard ratios) -> durée de vie et valeur attendues par segment

suppressMessages({ library(jsonlite); library(survival) })
root <- if (dir.exists("data")) "." else ".."
df <- read.csv(file.path(root, "data/telco.csv"), stringsAsFactors = FALSE)
df$event <- as.integer(df$Churn == "Yes")
df$time <- pmax(df$tenure, 0.5)            # ancienneté 0 = client arrivé ce mois-ci
# Un client toujours abonné est "censuré" : on sait seulement qu'il est resté au moins `tenure` mois

km_curve <- function(fit, strata_names) {
  s <- summary(fit, times = 0:72, extend = TRUE)
  st <- if (is.null(s$strata)) rep("Tous", length(s$time)) else as.character(s$strata)
  lapply(split(data.frame(t = s$time, s = round(s$surv * 100, 2), lo = round(s$lower * 100, 2),
                          hi = round(s$upper * 100, 2), g = st), st),
         function(d) list(t = d$t, s = d$s, lo = d$lo, hi = d$hi))
}

# ---------- 1. Kaplan-Meier global et par contrat ----------
fit_all <- survfit(Surv(time, event) ~ 1, data = df)
fit_ct <- survfit(Surv(time, event) ~ Contract, data = df)
lr_ct <- survdiff(Surv(time, event) ~ Contract, data = df)
p_ct <- 1 - pchisq(lr_ct$chisq, length(lr_ct$n) - 1)
fit_int <- survfit(Surv(time, event) ~ InternetService, data = df)
fit_pay <- survfit(Surv(time, event) ~ PaymentMethod, data = df)

surv_at <- function(fit, t) { s <- summary(fit, times = t, extend = TRUE)
  setNames(round(s$surv * 100, 1), if (is.null(s$strata)) "Tous" else sub(".*=", "", as.character(s$strata))) }

# Durée de vie moyenne restreinte à 72 mois (RMST) = aire sous la courbe de survie
rmst <- function(fit) { tab <- summary(fit, rmean = 72)$table
  if (is.null(dim(tab))) c(Tous = unname(tab["rmean"])) else setNames(round(tab[, "rmean"], 1), sub(".*=", "", rownames(tab))) }

# ---------- 2. Modèle de Cox ----------
df$Contract <- relevel(factor(df$Contract), ref = "Month-to-month")
df$InternetService <- relevel(factor(df$InternetService), ref = "DSL")
df$PaymentMethod <- relevel(factor(df$PaymentMethod), ref = "Mailed check")
cox <- coxph(Surv(time, event) ~ Contract + InternetService + PaymentMethod + TechSupport + OnlineSecurity +
               SeniorCitizen + Partner + PaperlessBilling + MonthlyCharges, data = df)
sc <- summary(cox)
hr <- data.frame(var = rownames(sc$conf.int), hr = round(sc$conf.int[, 1], 2),
                 lo = round(sc$conf.int[, 3], 2), hi = round(sc$conf.int[, 4], 2),
                 p = signif(sc$coefficients[, 5], 2))
hr <- hr[!is.na(hr$hr), ]; hr <- hr[order(hr$hr), ]; rownames(hr) <- NULL
cindex <- round(sc$concordance[1], 3)
ph <- cox.zph(cox)$table
ph_global <- round(ph["GLOBAL", "p"], 4)

# ---------- 3. Valeur attendue par contrat ----------
mrr <- tapply(df$MonthlyCharges, df$Contract, mean)
life <- rmst(fit_ct)
valeur <- data.frame(contrat = names(life), duree = unname(life), mrr = round(as.numeric(mrr[names(life)]), 2))
valeur$valeur <- round(valeur$duree * valeur$mrr)

out <- list(
  kpi = list(clients = nrow(df), evenements = sum(df$event), censures = sum(df$event == 0),
             survie_12 = unname(surv_at(fit_all, 12)), survie_72 = unname(surv_at(fit_all, 72)),
             rmst = unname(rmst(fit_all)), cindex = unname(cindex), p_logrank = signif(p_ct, 3), ph_global = ph_global),
  km_all = km_curve(fit_all), km_contrat = km_curve(fit_ct), km_internet = km_curve(fit_int), km_paiement = km_curve(fit_pay),
  survie_12_contrat = as.list(surv_at(fit_ct, 12)), survie_24_contrat = as.list(surv_at(fit_ct, 24)),
  hr = hr, valeur = valeur
)
write_json(out, file.path(root, "results/r_survie.json"), auto_unbox = TRUE, digits = 6, pretty = TRUE)
print(out$kpi); print(hr); print(valeur); print(out$survie_12_contrat)
