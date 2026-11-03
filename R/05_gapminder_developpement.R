# Projet : Richesse et espérance de vie dans le monde, 1952-2007 (l'analyse rendue célèbre par Hans Rosling)
# Données : gapminder (142 pays × 12 années : espérance de vie, population, PIB par habitant), package R de Jenny Bryan
# Méthode : modèle log-linéaire par année (lm) -> la « courbe de Preston » se déplace-t-elle ?
#           -> convergence entre continents (écart-type, coefficient de variation)
#           -> pays atypiques (résidus) -> chocs (baisses d'espérance de vie)

suppressMessages(library(jsonlite))
root <- if (dir.exists("data")) "." else ".."
g <- read.delim(file.path(root, "data/gapminder.tsv"), stringsAsFactors = FALSE)
g$lgdp <- log10(g$gdpPercap)
years <- sort(unique(g$year))

# ---------- 1. Un modèle par année : espérance de vie ~ log10(PIB/hab) ----------
fits <- lapply(years, function(y) lm(lifeExp ~ lgdp, data = g[g$year == y, ]))
coefs <- data.frame(annee = years,
                    pente = sapply(fits, function(m) round(coef(m)[2], 2)),
                    constante = sapply(fits, function(m) round(coef(m)[1], 2)),
                    r2 = sapply(fits, function(m) round(summary(m)$r.squared, 3)))
# Espérance de vie prédite pour un pays à 1 000 $ et à 10 000 $ / habitant, chaque année
coefs$a_1000 <- round(coefs$constante + coefs$pente * 3, 1)
coefs$a_10000 <- round(coefs$constante + coefs$pente * 4, 1)

# Modèle global avec effet année et continent
m_all <- lm(lifeExp ~ lgdp + factor(year) + continent, data = g)
s_all <- summary(m_all)

# ---------- 2. Convergence ----------
conv <- aggregate(cbind(lifeExp, gdpPercap) ~ year, g, function(x) c(moy = mean(x), sd = sd(x), cv = sd(x) / mean(x)))
conv <- data.frame(annee = conv$year, le_moy = round(conv$lifeExp[, "moy"], 1), le_sd = round(conv$lifeExp[, "sd"], 2),
                   pib_cv = round(conv$gdpPercap[, "cv"], 3))
cont <- aggregate(lifeExp ~ continent + year, g, function(x) round(weighted.mean(x), 1))
cont_w <- do.call(rbind, lapply(split(g, list(g$continent, g$year)), function(d)
  data.frame(continent = d$continent[1], year = d$year[1], le = round(weighted.mean(d$lifeExp, d$pop), 1))))
cont_w <- cont_w[order(cont_w$continent, cont_w$year), ]

# ---------- 3. Pays atypiques en 2007 ----------
d07 <- g[g$year == 2007, ]
d07$res <- round(resid(fits[[length(fits)]]), 1)
atyp <- d07[order(d07$res), c("country", "continent", "lifeExp", "gdpPercap", "res")]
atyp_bas <- head(atyp, 8); atyp_haut <- tail(atyp[order(atyp$res), ], 8)

# ---------- 4. Chocs : plus fortes baisses d'espérance de vie sur 5 ans ----------
g <- g[order(g$country, g$year), ]
g$delta <- ave(g$lifeExp, g$country, FUN = function(x) c(NA, diff(x)))
chocs <- head(g[order(g$delta), c("country", "continent", "year", "lifeExp", "delta")], 10)
chocs$delta <- round(chocs$delta, 1)

# ---------- Données pour le graphique animé ----------
bubbles <- lapply(years, function(y) { d <- g[g$year == y, ]
  list(year = y, c = d$country, k = d$continent, x = round(d$gdpPercap), y = round(d$lifeExp, 1), p = round(d$pop / 1e6, 2)) })

out <- list(
  kpi = list(pays = length(unique(g$country)), annees = length(years), gain_vie = round(mean(g$lifeExp[g$year == 2007]) - mean(g$lifeExp[g$year == 1952]), 1),
             r2_2007 = coefs$r2[length(years)], r2_global = round(s_all$r.squared, 3),
             gain_x10 = coefs$pente[length(years)], decalage = round(coefs$a_1000[length(years)] - coefs$a_1000[1], 1)),
  coefs = coefs, conv = conv, continents = cont_w, atyp_bas = atyp_bas, atyp_haut = atyp_haut, chocs = chocs, bubbles = bubbles
)
write_json(out, file.path(root, "results/r_gapminder.json"), auto_unbox = TRUE, digits = 6, pretty = FALSE)
print(out$kpi); print(coefs); print(atyp_bas); print(atyp_haut); print(chocs)
