# soil_sample_signatures
This is a python script I used to employ example data mining tools from scikit-learn to identify spatial variability in petroleum hydrocarbon product signatures in soil samples collected from a large environmental site. It serves as a simple demo as to how to use both pandas and scikit-learn to easily sort through a medium-sized data set and glean insights.

The following tab-delimited input files are required:

* fuel_ref.txt = reference compositions for petroleum products (by TPH range); several examples are included for each product so that averages and standard deviations can be computed
* survey.txt = soil boring survey information (location ID, northing, easting, and surface elevation); example file not currently provided (to protect anonymity of the site)
* soil_samples.txt = soil chemistry data (location ID that matching survey data file, depth, date, name of analyte, result); example file not currently provided (to protect anonymity of the site)

More background information can be found here: https://numericalenvironmental.wordpress.com/2016/08/17/classifiers-for-distributed-soil-samples/

Email me with questions at walt.mcnab@gmail.com. 

THIS CODE/SOFTWARE IS PROVIDED IN SOURCE OR BINARY FORM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
