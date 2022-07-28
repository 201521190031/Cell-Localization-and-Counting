# include <mln/core/image/image2d.hh>
# include <mln/value/int_u8.hh>
# include <mln/data/fill.hh>
# include <mln/io/pgm/all.hh>
# include <mln/io/ppm/all.hh>
# include <mln/pw/all.hh>
# include <mln/data/paste.hh>
# include <mln/core/site_set/p_set.hh>
# include <mln/data/stretch.hh>
# include <mln/data/convert.hh>
# include <mln/labeling/colorize.hh>
# include <mln/literal/colors.hh>
# include <mln/labeling/regional_maxima.hh>
# include <mln/labeling/regional_minima.hh>
# include <mln/core/routine/duplicate.hh>
#include <mln/core/alias/neighb2d.hh>
# include <iomanip>
# include <fstream>
# include <ostream>
#include <sstream>

void usage(char* argv[])
{
  std::cout << argv[0] <<" density.pgm thr in.pgm out.ppm" << std::endl;
  abort();
}


int main(int argc, char *argv[])
{
  if (argc != 5)
    usage(argv);
  using namespace mln;

  image2d<value::int_u8> f;
  io::pgm::load(f, argv[1]);

  unsigned thr = std::atoi(argv[2]);

  mln_piter_(image2d<value::int_u8>) p(f.domain());
  for_all(p)
  {
    if(f(p) > thr)
      f(p) = 255;
  }


  unsigned nMin = 0;
  image2d<unsigned> labMin = labeling::regional_minima(f, c4(), nMin);
  //std::cout << "nMin = " << nMin << std::endl;
  //std::cout << labMin.at_(0, 0) << std::endl;

  image2d<value::int_u8> inOut;
  io::pgm::load(inOut, argv[3]);
  image2d<value::rgb8> out = data::convert(value::rgb8(), inOut);

  for_all(p)
  {
    if(labMin(p) > 0)
      out(p) = value::rgb8(255, 0, 0);
  }
  io::ppm::save(out, argv[4]);

}
