# Projet : Segmenter une base clients et cibler la prochaine campagne marketing
# Données : base clients d'un distributeur alimentaire (cas iFood, 2 240 clients, 29 variables)
# Méthode : nettoyage -> variables RFM -> k-means (choix de k : coude + silhouette)
#           -> profils des segments -> régression logistique de réponse -> ciblage rentable

suppressMessages(library(jsonlite))
library(cluster)
root <- if (dir.exists("data")) "." else ".."
df <- read.csv(file.path(root, "data/marketing.csv"))

# ---------- 1. Nettoyage ----------
n0 <- nrow(df)
df <- df[!is.na(df$Income), ]                       # 24 revenus manquants
df <- df[df$Year_Birth >= 1930 & df$Income < 600000, ] # âges et revenus aberrants
df$Marital_Status <- ifelse(df$Marital_Status %in% c("Married", "Together"), "En couple", "Seul")  # "YOLO", "Absurd"... regroupés
df$Education <- ifelse(df$Education %in% c("Basic", "2n Cycle"), "Base", df$Education)
df$Age <- 2014 - df$Year_Birth                     # base extraite en 2014
df$Dt_Customer <- as.Date(df$Dt_Customer)
df$Anciennete <- as.numeric(as.Date("2014-12-31") - df$Dt_Customer) / 30.4
mnt <- grep("^Mnt", names(df), value = TRUE)
df$Depense <- rowSums(df[, mnt])
df$Frequence <- df$NumWebPurchases + df$NumCatalogPurchases + df$NumStorePurchases
df$Enfants <- df$Kidhome + df$Teenhome
df$CampAcc <- rowSums(df[, grep("^AcceptedCmp", names(df))])
df$PartWeb <- with(df, ifelse(Frequence > 0, NumWebPurchases / Frequence, 0))

# ---------- 2. Clustering ----------
vars <- c("Recency", "Frequence", "Depense", "Income", "Enfants", "NumDealsPurchases", "PartWeb")
X <- scale(cbind(df[, vars[1:2]], log1p(df$Depense), df[, vars[4:7]]))
set.seed(42)
ks <- 2:8
wss <- sapply(ks, function(k) kmeans(X, k, nstart = 25, iter.max = 50)$tot.withinss)
idx <- sample(nrow(X), 1200)
D <- dist(X[idx, ])
sil <- sapply(ks, function(k) {
  km <- kmeans(X, k, nstart = 25, iter.max = 50)
  mean(silhouette(km$cluster[idx], D)[, 3])
})
k <- 4                                               # compromis coude / lisibilité métier
km <- kmeans(X, k, nstart = 50, iter.max = 100)
df$Segment <- km$cluster

prof <- aggregate(cbind(Income, Depense, Frequence, Recency, Enfants, NumDealsPurchases,
                        PartWeb, Response, Age) ~ Segment, df, mean)
prof$Taille <- as.vector(table(df$Segment))
prof$PartCA <- as.vector(round(tapply(df$Depense, df$Segment, sum) / sum(df$Depense) * 100, 1))
# Nommage automatique des segments à partir de leur profil
noms <- character(k)
i_prem <- which.max(prof$Depense)
i_promo <- setdiff(order(prof$NumDealsPurchases, decreasing = TRUE), i_prem)[1]
reste <- setdiff(seq_len(k), c(i_prem, i_promo))
i_dort <- reste[which.max(prof$Recency[reste])]
noms[i_prem] <- "Premium"
noms[i_promo] <- "Familles chasseuses de promos"
noms[i_dort] <- "Clients endormis"
noms[noms == ""] <- "Petits paniers actifs"
prof$Nom <- noms
df$NomSegment <- noms[df$Segment]

# PCA pour visualiser les segments en 2D
pc <- prcomp(X)
var_exp <- round(summary(pc)$importance[2, 1:2] * 100, 1)
set.seed(1); s <- sample(nrow(df), 700)
scatter <- data.frame(x = round(pc$x[s, 1], 2), y = round(pc$x[s, 2], 2), seg = df$NomSegment[s])

# ---------- 3. Modèle de réponse à la campagne ----------
set.seed(7)
tr <- sample(nrow(df), 0.7 * nrow(df))
f <- Response ~ Recency + log1p(Depense) + Income + Enfants + NumDealsPurchases + NumWebVisitsMonth +
  CampAcc + Anciennete + PartWeb + Education + Marital_Status
df$Education <- factor(df$Education); df$Marital_Status <- factor(df$Marital_Status)
m <- glm(f, data = df[tr, ], family = binomial)
te <- df[-tr, ]
p <- predict(m, te, type = "response")
auc <- function(y, p) { r <- rank(p); n1 <- sum(y); n0 <- length(y) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
AUC <- auc(te$Response, p)
co <- summary(m)$coefficients
co <- co[rownames(co) != "(Intercept)", ]; co <- co[order(co[, 4]), ][1:8, ]
coefs <- data.frame(var = rownames(co), or = round(exp(co[, 1]), 2), p = signif(co[, 4], 2))

# Ciblage : coût d'un contact 3, revenu d'une réponse 11 (valeurs Z_CostContact / Z_Revenue du jeu)
cost <- 3; rev <- 11
ord <- order(p, decreasing = TRUE)
cum <- data.frame(part = seq_along(ord) / length(ord), rep = cumsum(te$Response[ord]))
cum$profit <- cum$rep * rev - seq_along(ord) * cost
best <- cum[which.max(cum$profit), ]
all_profit <- sum(te$Response) * rev - nrow(te) * cost
gains <- cum[round(seq(1, nrow(cum), length.out = 21)), ]

out <- list(
  kpi = list(clients = nrow(df), retires = n0 - nrow(df), taux_reponse = round(mean(df$Response) * 100, 1),
             auc = round(AUC, 3), k = k, sil = round(sil[ks == k], 3)),
  elbow = list(k = ks, wss = round(wss), sil = round(sil, 3)),
  profils = prof, scatter = scatter, var_exp = var_exp, coefs = coefs,
  ciblage = list(part = round(best$part * 100, 1), profit = best$profit, profit_tous = all_profit,
                 rep_captees = round(best$rep / sum(te$Response) * 100, 1)),
  gains = list(part = round(gains$part * 100), rep = round(gains$rep / sum(te$Response) * 100, 1))
)
write_json(out, file.path(root, "results/r_segmentation.json"), auto_unbox = TRUE, digits = 6, pretty = TRUE)
print(prof); print(out$kpi); print(out$ciblage); print(coefs); print(round(sil, 3))
