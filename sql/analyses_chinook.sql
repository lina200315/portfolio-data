-- =====================================================================
-- Projet : Analyse commerciale d'une boutique de musique en ligne (SQL)
-- Base   : Chinook (SQLite) — 11 tables : clients, factures, lignes, titres, genres, employés...
-- Objectif : construire les KPI d'un comité de direction uniquement en SQL
-- Notions : jointures multiples, CTE, fonctions de fenêtre (LAG, NTILE, RANK,
--           SUM OVER), CASE, agrégations conditionnelles, analyse de cohortes, RFM
-- =====================================================================

-- @name: q01_kpi
-- @title: KPI globaux
SELECT
    COUNT(DISTINCT i.CustomerId)                         AS clients,
    COUNT(DISTINCT i.InvoiceId)                          AS commandes,
    ROUND(SUM(i.Total), 2)                               AS ca_total,
    ROUND(SUM(i.Total) / COUNT(DISTINCT i.InvoiceId), 2) AS panier_moyen,
    ROUND(SUM(i.Total) / COUNT(DISTINCT i.CustomerId), 2) AS ca_par_client,
    ROUND(1.0 * COUNT(DISTINCT i.InvoiceId) / COUNT(DISTINCT i.CustomerId), 1) AS commandes_par_client,
    (SELECT SUM(Quantity) FROM InvoiceLine)              AS titres_vendus
FROM Invoice i;

-- @name: q02_mensuel
-- @title: CA mensuel, croissance vs N-1 et moyenne mobile 3 mois
WITH m AS (
    SELECT strftime('%Y-%m', InvoiceDate) AS mois, SUM(Total) AS ca
    FROM Invoice GROUP BY 1
)
SELECT mois,
       ROUND(ca, 2) AS ca,
       ROUND(AVG(ca) OVER (ORDER BY mois ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS mm3,
       ROUND(100.0 * (ca - LAG(ca, 12) OVER (ORDER BY mois)) / LAG(ca, 12) OVER (ORDER BY mois), 1) AS croissance_n1,
       ROUND(SUM(ca) OVER (ORDER BY mois), 2) AS ca_cumule
FROM m ORDER BY mois;

-- @name: q03_pays
-- @title: Pays : CA, clients, panier moyen et part du CA (Pareto)
WITH p AS (
    SELECT c.Country AS pays, COUNT(DISTINCT c.CustomerId) AS clients,
           COUNT(i.InvoiceId) AS commandes, SUM(i.Total) AS ca
    FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId
    GROUP BY c.Country
)
SELECT pays, clients, commandes, ROUND(ca, 2) AS ca,
       ROUND(ca / commandes, 2) AS panier_moyen,
       ROUND(100.0 * ca / SUM(ca) OVER (), 1) AS part_ca,
       ROUND(100.0 * SUM(ca) OVER (ORDER BY ca DESC) / SUM(ca) OVER (), 1) AS part_cumulee
FROM p ORDER BY ca DESC;

-- @name: q04_genres
-- @title: Genres musicaux : CA, volume et rang
SELECT g.Name AS genre,
       SUM(il.Quantity) AS titres_vendus,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS ca,
       COUNT(DISTINCT i.CustomerId) AS clients,
       RANK() OVER (ORDER BY SUM(il.UnitPrice * il.Quantity) DESC) AS rang
FROM InvoiceLine il
JOIN Track t   ON t.TrackId = il.TrackId
JOIN Genre g   ON g.GenreId = t.GenreId
JOIN Invoice i ON i.InvoiceId = il.InvoiceId
GROUP BY g.Name ORDER BY ca DESC;

-- @name: q05_catalogue
-- @title: Catalogue dormant : part des titres jamais vendus par genre
SELECT g.Name AS genre,
       COUNT(*) AS titres_catalogue,
       SUM(CASE WHEN il.TrackId IS NULL THEN 1 ELSE 0 END) AS jamais_vendus,
       ROUND(100.0 * SUM(CASE WHEN il.TrackId IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS part_dormante
FROM Track t
JOIN Genre g ON g.GenreId = t.GenreId
LEFT JOIN (SELECT DISTINCT TrackId FROM InvoiceLine) il ON il.TrackId = t.TrackId
GROUP BY g.Name HAVING COUNT(*) >= 50
ORDER BY part_dormante DESC;

-- @name: q06_vendeurs
-- @title: Performance des conseillers commerciaux
SELECT e.FirstName || ' ' || e.LastName AS conseiller,
       e.Title AS poste,
       COUNT(DISTINCT c.CustomerId) AS portefeuille_clients,
       ROUND(SUM(i.Total), 2) AS ca,
       ROUND(SUM(i.Total) / COUNT(DISTINCT c.CustomerId), 2) AS ca_par_client,
       ROUND(100.0 * SUM(i.Total) / SUM(SUM(i.Total)) OVER (), 1) AS part_ca
FROM Employee e
JOIN Customer c ON c.SupportRepId = e.EmployeeId
JOIN Invoice i  ON i.CustomerId = c.CustomerId
GROUP BY e.EmployeeId ORDER BY ca DESC;

-- @name: q07_rfm
-- @title: Segmentation RFM des clients (NTILE en quartiles)
WITH base AS (
    SELECT CustomerId,
           julianday((SELECT MAX(InvoiceDate) FROM Invoice)) - julianday(MAX(InvoiceDate)) AS recence_j,
           COUNT(*) AS frequence,
           SUM(Total) AS montant
    FROM Invoice GROUP BY CustomerId
), scores AS (
    SELECT *,
           NTILE(4) OVER (ORDER BY recence_j DESC) AS r,
           NTILE(4) OVER (ORDER BY frequence)     AS f,
           NTILE(4) OVER (ORDER BY montant)       AS m
    FROM base
)
SELECT CASE
         WHEN r >= 3 AND m >= 3 THEN 'Champions'
         WHEN r <= 2 AND m >= 3 THEN 'À réactiver (gros clients)'
         WHEN r >= 3 AND m <= 2 THEN 'Actifs petits paniers'
         ELSE 'Faible valeur' END AS segment,
       COUNT(*) AS clients,
       ROUND(AVG(recence_j), 0) AS recence_moy_j,
       ROUND(AVG(montant), 2) AS montant_moy,
       ROUND(SUM(montant), 2) AS ca
FROM scores GROUP BY 1 ORDER BY ca DESC;

-- @name: q08_cohortes
-- @title: Cohortes : part des clients de chaque cohorte qui rachètent dans les trimestres suivants
WITH first AS (
    SELECT CustomerId, MIN(InvoiceDate) AS d0 FROM Invoice GROUP BY CustomerId
), act AS (
    SELECT DISTINCT i.CustomerId,
           strftime('%Y', f.d0) AS cohorte,
           ((CAST(strftime('%Y', i.InvoiceDate) AS INT) - CAST(strftime('%Y', f.d0) AS INT)) * 12
             + CAST(strftime('%m', i.InvoiceDate) AS INT) - CAST(strftime('%m', f.d0) AS INT)) / 3 AS trimestre
    FROM Invoice i JOIN first f ON f.CustomerId = i.CustomerId
)
SELECT cohorte, trimestre,
       COUNT(*) AS clients_actifs,
       ROUND(100.0 * COUNT(*) / FIRST_VALUE(COUNT(*)) OVER (PARTITION BY cohorte ORDER BY trimestre), 1) AS retention
FROM act WHERE trimestre <= 8
GROUP BY cohorte, trimestre ORDER BY cohorte, trimestre;

-- @name: q09_top_clients
-- @title: Top 10 clients, avec leur genre préféré (sous-requête corrélée)
WITH ca AS (
    SELECT c.CustomerId, c.FirstName || ' ' || c.LastName AS client, c.Country AS pays,
           ROUND(SUM(i.Total), 2) AS ca, COUNT(*) AS commandes
    FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId
    GROUP BY c.CustomerId
)
SELECT client, pays, ca, commandes,
       (SELECT g.Name FROM Invoice i2
          JOIN InvoiceLine il ON il.InvoiceId = i2.InvoiceId
          JOIN Track t ON t.TrackId = il.TrackId
          JOIN Genre g ON g.GenreId = t.GenreId
         WHERE i2.CustomerId = ca.CustomerId
         GROUP BY g.Name ORDER BY COUNT(*) DESC LIMIT 1) AS genre_prefere
FROM ca ORDER BY ca DESC LIMIT 10;

-- @name: q10_panier
-- @title: Ventes croisées : paires de genres achetés dans la même commande
WITH g AS (
    SELECT DISTINCT il.InvoiceId, gn.Name AS genre
    FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId JOIN Genre gn ON gn.GenreId = t.GenreId
)
SELECT a.genre AS genre_a, b.genre AS genre_b, COUNT(*) AS commandes_communes,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(DISTINCT InvoiceId) FROM g WHERE genre = a.genre), 1) AS pct_des_commandes_a
FROM g a JOIN g b ON a.InvoiceId = b.InvoiceId AND a.genre < b.genre
GROUP BY a.genre, b.genre ORDER BY commandes_communes DESC LIMIT 10;
