use super::{abs_diff, Clust, Event};
use std::cmp::min;
/*
 * this implementation has another Buffer and moves the Clusters out of it. The performance is
 * worse compared to the other implementation
 */
pub fn clust_analysis_with_ring(hits: &[Event], eps_space: u16, eps_time: f64) -> Vec<Clust> {
    let eps_time_count = (eps_time / 1e-12) as i64; // time is kept as int for faster computation
    let mut current_cluster = Vec::<Clust>::with_capacity(10);
    let mut extracted_cluster = Vec::<Clust>::with_capacity(hits.len() / 2);

    'outer: for hit in hits {
        let mut i = 0;
        while i < current_cluster.len() {
            let clust_is_current = { (hit.time - current_cluster[i].time).abs() <= eps_time_count };
            if !clust_is_current {
                let val = current_cluster.remove(i);
                extracted_cluster.push(val);
            } else {
                i += 1;
            }
        }
        for clust in &mut current_cluster {
            let close_to_cluster = abs_diff(hit.x, clust.x) + abs_diff(hit.y, clust.y) <= eps_space;
            if close_to_cluster {
                clust.size += 1;
                clust.sum += hit.intens;
                clust.time = min(hit.time, clust.time);
                if clust.intens < hit.intens {
                    clust.intens = hit.intens;
                    clust.x = hit.x;
                    clust.y = hit.y;
                }
                continue 'outer;
            }
        }
        let clust = Clust {
            x: hit.x,
            y: hit.y,
            time: hit.time,
            size: 1,
            sum: hit.intens,
            intens: hit.intens,
        };
        current_cluster.push(clust);
    }
    for clust in current_cluster {
        extracted_cluster.push(clust);
    }
    return extracted_cluster;
}
/*
 * Only check cluster until it gets current enough
 */
pub fn clust_analysis_skip(hits: &[Event], eps_space: u16, eps_time: f64) -> Vec<Clust> {
    let eps_time_count = (eps_time / 1e-12) as i64; // time is kept as int for faster computation

    let mut extracted_cluster = Vec::<Clust>::with_capacity(hits.len() / 2);

    let mut oldest_index = 0;
    'outer: for hit in hits {
        let mut check_time = true;
        for idx in oldest_index..extracted_cluster.len() {
            //let total_len= extracted_cluster.len();
            //for idx in total_len-4..total_len {
            let mut clust = &mut extracted_cluster[idx];
            if check_time {
                let clust_is_current = (hit.time - clust.time).abs() <= eps_time_count;
                if !clust_is_current {
                    oldest_index = idx;
                    continue;
                }
                check_time = false;
            }
            let close_to_cluster = abs_diff(hit.x, clust.x) + abs_diff(hit.y, clust.y) <= eps_space;
            if close_to_cluster {
                clust.size += 1;
                clust.sum += hit.intens;
                clust.time = min(hit.time, clust.time);
                if clust.intens < hit.intens {
                    clust.intens = hit.intens;
                    clust.x = hit.x;
                    clust.y = hit.y;
                }
                continue 'outer;
            }
        }
        let clust = Clust {
            x: hit.x,
            y: hit.y,
            time: hit.time,
            size: 1,
            sum: hit.intens,
            intens: hit.intens,
        };
        extracted_cluster.push(clust);
    }
    return extracted_cluster;
}
