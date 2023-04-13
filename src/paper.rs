/*
 * This is the example code from the paper.
 */

use std::cmp::min;
struct Hit {
    x: u16, y: u16, toa: i64, tot: u16,
}
struct Cluster {
    x: u16, y: u16, toa: i64, size: u16, summed_tot: u16, tot: u16,
}

/* Technical: Calculates abs(x-y) of unsigned integer variables */
fn abs_diff(x: u16, y: u16) -> u16 {
    return u16::saturating_sub(x, y) + u16::saturating_sub(y, x);
}

#[allow(unused)]
fn cluster_analysis(hits: &[Hit], eps_space: u16, eps_time: f64, b_offset: usize)
    -> Vec<Cluster> {
    let eps_time_count = (eps_time / 1e-12) as i64; //eps_time in (unit of camera)
    let mut extracted_cluster = Vec::<Cluster>::with_capacity(hits.len() / 2);
    'outer: for hit in hits {
        let start_idx = extracted_cluster.len().saturating_sub(b_offset);
        for cluster in &mut extracted_cluster[start_idx..] { // last b_offset clusters
            let cluster_is_current = (hit.toa - cluster.toa).abs() <= eps_time_count; 
            let cluster_is_close = abs_diff(hit.x, cluster.x) + 
                                   abs_diff(hit.y, cluster.y) <= eps_space; 
            if cluster_is_current && cluster_is_close {
                cluster.size += 1;
                cluster.summed_tot += hit.tot;
                cluster.toa = min(hit.toa, cluster.toa);
                if cluster.tot < hit.tot { // assign position to strongest hit
                    cluster.tot = hit.tot;
                    cluster.x = hit.x;
                    cluster.y = hit.y;
                }
                continue 'outer; // evaluate next hit
            }
        }
        let new_cluster = Cluster { // no matching cluster found -> new cluster
            x: hit.x, y: hit.y, toa: hit.toa, size: 1, 
            summed_tot: hit.tot, tot: hit.tot,
        };
        extracted_cluster.push(new_cluster);
    }
    return extracted_cluster;
}


