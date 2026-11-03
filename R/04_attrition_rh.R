# Projet : People analytics — pourquoi les salariés quittent-ils l'entreprise ?
# Données : IBM HR Analytics Employee Attrition (1 470 salariés, 35 variables, données fictives créées par IBM)
# Méthode : nettoyage (variables constantes) -> taux de départ par facteur -> tests du khi-deux
#           -> régression logistique (odds ratios) -> arbre de décision lisible (rpart) -> coût du turnover

suppressMessages({ library(jsonlite); library(rpart) })
root <- if (dir.exists("data")) "." else ".."
df <- read.csv(file.path(root, "data/hr_attrition.csv"), fileEncoding = "UTF-8-BOM", stringsAsFactors = TRUE)

# ---------- 1. Nettoyage ----------
const <- names(df)[sapply(df, function(x) length(unique(x)) == 1)]   # EmployeeCount, Over18, StandardHours
df <- df[, !(names(df) %in% c(const, "EmployeeNumber"))]
df$Depart <- as.integer(df$Attrition == "Yes")

# ---------- 2. Taux de départ par facteur ----------
rate <- function(v) {
  t <- aggregate(df$Depart, list(g = df[[v]]), function(x) c(taux = mean(x) * 100, n = length(x)))
  p <- suppressWarnings(chisq.test(table(df[[v]], df$Depart))$p.value)
  list(var = v, p = signif(p, 3), g = as.character(t$g), taux = round(t$x[, "taux"], 1), n = t$x[, "n"])
}
df$Tranche_age <- cut(df$Age, c(17, 25, 30, 35, 45, 60), labels = c("18-25", "26-30", "31-35", "36-45", "46-60"))
df$Tranche_anc <- cut(df$YearsAtCompany, c(-1, 1, 3, 5, 10, 40), labels = c("0-1 an", "2-3 ans", "4-5 ans", "6-10 ans", "> 10 ans"))
df$Tranche_sal <- cut(df$MonthlyIncome, quantile(df$MonthlyIncome, 0:5 / 5), include.lowest = TRUE,
                      labels = c("Q1 (bas)", "Q2", "Q3", "Q4", "Q5 (haut)"))
facteurs <- lapply(c("OverTime", "JobRole", "MaritalStatus", "BusinessTravel", "Tranche_age", "Tranche_anc",
                     "Tranche_sal", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "StockOptionLevel", "Department"), rate)

# ---------- 3. Régression logistique ----------
set.seed(11)
tr <- sample(nrow(df), 0.7 * nrow(df))
f <- Depart ~ OverTime + Age + MonthlyIncome + JobSatisfaction + EnvironmentSatisfaction + WorkLifeBalance +
  JobInvolvement + StockOptionLevel + YearsAtCompany + YearsSinceLastPromotion + NumCompaniesWorked +
  DistanceFromHome + BusinessTravel + MaritalStatus + JobLevel + TotalWorkingYears + RelationshipSatisfaction
m <- glm(f, data = df[tr, ], family = binomial)
te <- df[-tr, ]
p <- predict(m, te, type = "response")
auc <- function(y, p) { r <- rank(p); n1 <- sum(y); n0 <- length(y) - n1; (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
co <- summary(m)$coefficients[-1, ]
co <- co[co[, 4] < 0.05, ]
or <- data.frame(var = rownames(co), or = round(exp(co[, 1]), 3), p = signif(co[, 4], 2))
or <- or[order(-abs(log(or$or))), ]; rownames(or) <- NULL

# ---------- 4. Arbre de décision (règles lisibles par les RH) ----------
tree <- rpart(Depart ~ OverTime + Age + MonthlyIncome + JobSatisfaction + EnvironmentSatisfaction + StockOptionLevel +
                YearsAtCompany + JobLevel + MaritalStatus + BusinessTravel + WorkLifeBalance + TotalWorkingYears,
              data = df[tr, ], method = "class", parms = list(prior = c(.5, .5)),
              control = rpart.control(cp = 0.02, minbucket = 25, maxdepth = 4))
pt <- predict(tree, te)[, 2]
fr <- tree$frame
leaves <- which(fr$var == "<leaf>")
paths <- path.rpart(tree, as.numeric(rownames(fr)[leaves]), print.it = FALSE)
regles <- data.frame(regle = sapply(paths, function(x) paste(x[-1], collapse = " ET ")),
                     salaries = fr$n[leaves],
                     taux = round(sapply(as.numeric(rownames(fr)[leaves]), function(node) {
                       w <- tree$where; mean(df$Depart[tr][w == which(rownames(fr) == node)]) * 100 }), 1))
regles <- regles[order(-regles$taux), ]; rownames(regles) <- NULL

# ---------- 5. Coût du turnover (hypothèse : 6 mois de salaire par départ, ordre de grandeur courant) ----------
cout <- sum(df$MonthlyIncome[df$Depart == 1] * 6)
cout_ot <- sum(df$MonthlyIncome[df$Depart == 1 & df$OverTime == "Yes"] * 6)

out <- list(
  kpi = list(salaries = nrow(df), departs = sum(df$Depart), taux = round(mean(df$Depart) * 100, 1),
             retirees = const, auc_glm = round(auc(te$Depart, p), 3), auc_arbre = round(auc(te$Depart, pt), 3),
             cout = cout, cout_ot = cout_ot,
             taux_ot = round(mean(df$Depart[df$OverTime == "Yes"]) * 100, 1), taux_no_ot = round(mean(df$Depart[df$OverTime == "No"]) * 100, 1)),
  facteurs = facteurs, or = or, regles = regles,
  importance = as.list(round(tree$variable.importance / sum(tree$variable.importance) * 100, 1))
)
write_json(out, file.path(root, "results/r_rh.json"), auto_unbox = TRUE, digits = 6, pretty = TRUE)
print(out$kpi); print(or); print(regles); print(head(out$importance, 6))
